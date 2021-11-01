import torch
import torch.nn as nn
import torch.nn.functional as F
from models.ecbsr import ECBSR
from models.plainsr import PlainSR
from models.tf.plainsr import plainsr_tf
from torch.utils.data import DataLoader
import math
import argparse, yaml
import utils
import os
from tqdm import tqdm
import tensorflow.keras.layers as TF_Layers
from tensorflow.python.tools import freeze_graph
from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2
import tensorflow as tf
from tensorflow.compat.v1 import graph_util
from tensorflow.python.keras import backend as tf_keras_backend

parser = argparse.ArgumentParser(description='ECBSR convertor')

## yaml configuration files
parser.add_argument('--config', type=str, default=None, help = 'pre-config file for training')

## paramters for ecbsr
parser.add_argument('--scale', type=int, default=4, help = 'scale for sr network')
parser.add_argument('--colors', type=int, default=1, help = '1(Y channls of YCbCr), 3(RGB)')
parser.add_argument('--m_ecbsr', type=int, default=4, help = 'number of ecb')
parser.add_argument('--c_ecbsr', type=int, default=8, help = 'channels of ecb')
parser.add_argument('--idt_ecbsr', type=int, default=0, help = 'incorporate identity mapping in ecb or not')
parser.add_argument('--act_type', type=str, default='prelu', help = 'prelu, relu, splus, rrelu')
parser.add_argument('--pretrain', type=str, default=None, help = 'path of pretrained model')

parser.add_argument('--target_frontend', type=str, default='pb-ckpt', help = 'target front-end for inference engine, e.g. onnx/pb-ckpt/pb-1.x/pb-2.x/tflite-fp32')
parser.add_argument('--output_folder', type=str, default='./', help = 'output folder')

parser.add_argument('--is_dynamic_batches', type=int, default=0, help = 'dynamic batches or not')
parser.add_argument('--inp_n', type=int, default=1, help = 'batch size of input data')
parser.add_argument('--inp_c', type=int, default=1, help = 'channel size of input data')
parser.add_argument('--inp_h', type=int, default=270, help = 'height of input data')
parser.add_argument('--inp_w', type=int, default=480, help = 'width of input data')

if __name__ == '__main__':
    args = parser.parse_args()
    if args.config:
       opt = vars(args)
       yaml_args = yaml.load(open(args.config), Loader=yaml.FullLoader)
       opt.update(yaml_args)

    if not os.path.exists(args.output_folder):
        os.makedirs(args.output_folder)
    
    if args.target_frontend == 'pb-1.x':
        # necessary !!!
        tf.compat.v1.disable_eager_execution()

    device = torch.device('cpu')
    ## definitions of model, loss, and optimizer
    model_ecbsr = ECBSR(module_nums=args.m_ecbsr, channel_nums=args.c_ecbsr, with_idt=args.idt_ecbsr, act_type=args.act_type, scale=args.scale, colors=args.colors).to(device)
    model_plain = PlainSR(module_nums=args.m_ecbsr, channel_nums=args.c_ecbsr, act_type=args.act_type, scale=args.scale, colors=args.colors).to(device)
    
    if args.pretrain is not None:
        print("load pretrained model: {}!".format(args.pretrain))
        model_ecbsr.load_state_dict(torch.load(args.pretrain))
    else:
        raise ValueError('the pretrain path is invalud!')

    ## copy weights from ecbsr to plainsr
    depth = len(model_ecbsr.backbone)
    for d in range(depth):
        module = model_ecbsr.backbone[d]
        act_type = module.act_type
        RK, RB = module.rep_params()
        model_plain.backbone[d].conv3x3.weight.data = RK
        model_plain.backbone[d].conv3x3.bias.data = RB

        if act_type == 'relu':     pass
        elif act_type == 'linear': pass
        elif act_type == 'prelu':  model_plain.backbone[d].act.weight.data = module.act.weight.data
        else: raise ValueError('invalid type of activation!')

    ## convert model to onnx
    output_name = utils.cur_timestamp_str()
    if args.target_frontend == 'onnx':
        output_name = os.path.join(args.output_folder, output_name + '.onnx')
        batch_size = args.inp_n
        fake_x = torch.rand(batch_size, args.inp_c, args.inp_h, args.inp_w, requires_grad=False)

        dynamic_params = None
        if args.is_dynamic_batches:
            dynamic_params = {'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}

        torch.onnx.export(
            model_plain, 
            fake_x, 
            output_name, 
            export_params=True, 
            opset_version=10, 
            do_constant_folding=True, 
            input_names=['input'], 
            output_names=['output'],
            dynamic_axes=dynamic_params
        )

    elif args.target_frontend == 'pb-ckpt' or \
         args.target_frontend == 'pb-1.x'  or \
         args.target_frontend == 'pb-2.x'  or \
         args.target_frontend == 'tflite-fp32':
        # output_name = os.path.join(args.output_folder, output_name + '.pb')
        tf_raw_dir = os.path.join(args.output_folder, output_name )
        model_plain_tf = plainsr_tf(args.m_ecbsr, args.c_ecbsr, args.act_type, args.scale, args.colors, args.inp_h, args.inp_w)

        depth  = len(model_plain.backbone)
        tf_idx = 0
        for d in range(depth):
            tf_idx += 1
            module = model_plain.backbone[d]
            act_type = module.act_type
            ## update weights of conv3x3
            K, B = module.conv3x3.weight, module.conv3x3.bias
            K, B = K.detach().numpy(), B.detach().numpy()
            RK_tf, RB_tf = K.transpose([2, 3, 1, 0]), B
            wgt_tf = [RK_tf, RB_tf]
            model_plain_tf.layers[tf_idx].set_weights(wgt_tf)
            ## update weights of activation
            if act_type == 'linear':
                pass
            elif act_type == 'relu':
                tf_idx += 1
            elif act_type == 'prelu':
                tf_idx += 1
                slope = module.act.weight.data
                slope = slope.view((1,1,-1))
                slope = slope.detach().numpy()
                slope_tf = slope
                wgt_tf = [slope_tf]
                model_plain_tf.layers[tf_idx].set_weights(wgt_tf)
            else:
                raise ValueError('invalid type of activation!')
        
        if args.target_frontend == 'pb-ckpt':
            # save checkpoints
            model_plain_tf.save(tf_raw_dir, overwrite=True, include_optimizer=False, save_format='tf')
        if args.target_frontend == 'tflite-fp32':
             # save checkpoints
            model_plain_tf.save(tf_raw_dir, overwrite=True, include_optimizer=False, save_format='tf')        
            # # Load trained SavedModel
            model = tf.saved_model.load(tf_raw_dir)
            # Setup fixed input shape
            input_shape = [1, args.inp_h, args.inp_w, args.inp_c]
            concrete_func = model.signatures[tf.saved_model.DEFAULT_SERVING_SIGNATURE_DEF_KEY]
            concrete_func.inputs[0].set_shape(input_shape)
            # Get tf.lite converter instance
            converter = tf.lite.TFLiteConverter.from_concrete_functions([concrete_func])
            # Use full integer operations in quantized model
            # converter.optimizations = [tf.lite.Optimize.DEFAULT]
            converter.optimizations = [tf.lite.Optimize.OPTIMIZE_FOR_LATENCY]  
            converter.inference_input_type = tf.float32
            converter.inference_output_type = tf.float32
            tflite_model = converter.convert()
            open('{}/model.tflite'.format(tf_raw_dir), 'wb').write(tflite_model) 
        elif args.target_frontend == 'pb-1.x':
            # save pb, tensorflow-1.x
            with tf_keras_backend.get_session() as sess:
                output_names = [out.op.name for out in model_plain_tf.outputs]
                input_graph_def = sess.graph.as_graph_def()
                for node in input_graph_def.node:
                    node.device = ""
                graph = graph_util.remove_training_nodes(input_graph_def)
                graph_frozen = graph_util.convert_variables_to_constants(sess, graph, output_names)
                tf.io.write_graph(
                    graph_or_graph_def=graph_frozen, 
                    logdir=tf_raw_dir, 
                    name='model.pb', 
                    as_text=False
                )
        elif args.target_frontend == 'pb-2.x':

            # Get frozen ConcreteFunction
            full_model = tf.function(lambda x: model_plain_tf(x))
            full_model = full_model.get_concrete_function([tf.TensorSpec(model_input.shape, model_input.dtype) for model_input in model_plain_tf.inputs])
            frozen_func = convert_variables_to_constants_v2(full_model)
            frozen_func.graph.as_graph_def()

            tf.io.write_graph(
                graph_or_graph_def=frozen_func.graph, 
                logdir=tf_raw_dir,
                name="model.pb",
                as_text=False
            )
    else:
        raise ValueError('invalid type of frontend!')
    