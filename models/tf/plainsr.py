import sys
import tensorflow as tf
import h5py
import math
import numpy as np
import cv2
from tensorflow.keras.layers import Conv2D, Conv2DTranspose, Input, PReLU, ReLU, UpSampling2D, Lambda
from tensorflow.keras.models import Model

import os
os.environ["CUDA_VISIBLE_DEVICES"]="-1"

def tf_conv3x3(inp, out_channels, act_type):
    y = Conv2D(filters=out_channels, kernel_size=3, strides=1, padding='same')(inp)
    if act_type == 'relu':
        y = ReLU()(y)
    elif act_type == 'prelu':
        y = PReLU(shared_axes=[1,2])(y)
    elif act_type == 'linear':
        pass
    else:
        raise ValueError('invalid act-type for tensorflow!')
    return y

def plainsr_tf(module_nums, channel_nums, act_type, scale, colors, input_h, input_w):
    inp = Input(shape=(input_h, input_w, colors))
    ## head
    y = tf_conv3x3(inp, channel_nums, act_type)
    ## body
    for i in range(module_nums):
        y = tf_conv3x3(y, channel_nums, act_type)
    if colors == 1:
        ## tail
        y = tf_conv3x3(y, colors*scale*scale, 'linear')
        y = y + inp
        # y = tf.clip_by_value(y, 0.0, 255.0)
        # y = tf.clip_by_value(y, 0.0, 1.0)
        ## upscaling
        out = tf.nn.depth_to_space(y, scale, data_format='NHWC')
    # if colors == 1:
    #     y = tf_conv3x3(y, colors*scale*scale, 'linear')
    #     out = tf.nn.depth_to_space(y, scale, data_format='NHWC') + tf.keras.layers.UpSampling2D(size=(scale, scale), data_format=None, interpolation='nearest')(inp)
    #     out = tf.clip_by_value(out, 0.0, 1.0)
    elif colors == 3:
        ## since internal data layout bwtween pytorch and tensorflow are quite different, e.g. NCHW for pytorch, NHWC for tensorflow
        ## input data layout of pixel-shuffle needs to be carefully handled
        
        ## tail
        y = tf_conv3x3(y, colors*scale*scale, 'linear')
        
        ## rgb layout
        r,g,b = tf.split(y, num_or_size_splits=colors, axis=3)
        ## upsaling
        tf_r, tf_g, tf_b = tf.split(y, num_or_size_splits=colors, axis=3)

        tf_r += r
        tf_g += g
        tf_b += b

        tf_r = tf.nn.depth_to_space(tf_r, scale, data_format='NHWC')
        tf_r = tf.clip_by_value(tf_r, 0.0, 255.0)
        tf_g = tf.nn.depth_to_space(tf_g, scale, data_format='NHWC')
        tf_g = tf.clip_by_value(tf_g, 0.0, 255.0)
        tf_b = tf.nn.depth_to_space(tf_b, scale, data_format='NHWC')
        tf_b = tf.clip_by_value(tf_b, 0.0, 255.0)
        out  = tf.concat(values=[tf_r, tf_g, tf_b], axis=3)
    else:
        raise ValueError('invalid colors!')
    return Model(inputs=inp, outputs=out)

if __name__ == '__main__':
    model_tf = plainsr_tf(module_nums=4, channel_nums=8, act_type='relu', scale=2, colors=3)
    for idx, layer in enumerate(model_tf.layers):
        wgt = layer.get_weights()
        nums = len(wgt)
        print(layer, idx)
        if nums == 1:
            pass
            # print(layer, isinstance(layer, PReLU))