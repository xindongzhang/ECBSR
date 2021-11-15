# Codes for ECBSR

>[Edge-oriented Convolution Block for Real-time Super Resolution on Mobile Devices](https://www4.comp.polyu.edu.hk/~cslzhang/paper/MM21_ECBSR.pdf) \
> Xindong Zhang, Hui Zeng, Lei Zhang \
> ACM Multimedia 2021


## Codes

An older version implemented based on [EDSR](https://github.com/sanghyun-son/EDSR-PyTorch) is place on **/legacy** folder. For more details, please refer to [/legacy/README.md](https://github.com/xindongzhang/ECBSR/tree/main/legacy/README.md). The following is the lighten version implemented by us.

### Dependencies & Installation

Please refer to the following simple steps for installation.

```
git clone https://github.com/xindongzhang/ECBSR.git
cd ECBSR
pip install -r requirements.txt
```

Training and benchmarking data can be downloaded from [DIV2K](https://cv.snu.ac.kr/research/EDSR/DIV2K.tar) and [benchmark](https://cv.snu.ac.kr/research/EDSR/benchmark.tar), respectively. Thanks for excellent work by EDSR.

### Training & Testing
You could also try less/larger batch-size, if there are limited/enough hardware resources in your GPU-server. ECBSR is trained and tested with colors=1, e.g Y channel out of Ycbcr.
```
cd ECBSR

## ecbsr-m4c8-x2-prelu(you can revise the parameters of the yaml-config file accordding to your environments)
python train.py --config ./configs/ecbsr_x2_m4c8_prelu.yml

## ecbsr-m4c8-x4-prelu
python train.py --config ./configs/ecbsr_x4_m4c8_prelu.yml

## ecbsr-m4c16-x2-prelu
python train.py --config ./configs/ecbsr_x2_m4c16_prelu.yml

## ecbsr-m4c16-x4-prelu
python train.py --config ./configs/ecbsr_x4_m4c16_prelu.yml
```

## Hardware deployment


### Frontend conversion

We provide [convertor](https://github.com/xindongzhang/ECBSR/blob/main/convert.py) for model conversion to different frontend, e.g. onnx/pb/tflite. We currently developed and tested the model with only one-channel(Y out of Ycbcr). Since the internal data-layout are quite different between tf(NHWC) and pytorch(NCHW), espetially for the pixelshuffle operation. Care must be taken to handle the data-layout, if you want to extend the pytorch-based training framework to RGB input data and deploy it on tensorflow. Follow are the demo scripts for model conversion to specific frontend:

```
## convert the trained pytorch model to onnx with plain-topology.
python convert.py --config xxx.yml --target_frontend onnx --output_folder XXX --inp_n 1 --inp_c 1 --inp_h 270 --inp_w 480

## convert the trained pytorch model to pb-1.x with plain-topology.
python convert.py --config xxx.yml --target_frontend pb-1.x --output_folder XXX --inp_n 1 --inp_c 1 --inp_h 270 --inp_w 480

## convert the trained pytorch model to pb-ckpt with plain-topology
python convert.py --config xxx.yml --target_frontend pb-ckpt --output_folder XXX --inp_n 1 --inp_c 1 --inp_h 270 --inp_w 480
```

### AI-Benchmark

You can download the newest version of evaluation tool from [AI-Benchmark](https://www.google.com.hk/url?sa=t&rct=j&q=&esrc=s&source=web&cd=&ved=2ahUKEwi5nIj7iMryAhVEZd4KHX5dCpIQFnoECAsQAw&url=https%3A%2F%2Fai-benchmark.com%2F&usg=AOvVaw3uZGyMiu_MMWy5_cLGpH8N). Then you can install the app via ADB tools,

```
adb install -r [name-of-ai-benchmar].apk
```

### MNN (Come soon!)

For universal CPU & GPU of mobile hardware implementation.

### RKNN (Come soon!)

For NPU inplementation of Rockchip hardware, e.g. RK3399Pro/RK1808.

### MiniNet (Come soon!)

A super light-weight CNN inference framework implemented by us, with only conv-3x3, element-wise op, ReLU(PReLU) activations, and pixel-shuffle for common super resolution task. For more details, please refer to /ECBSR/deploy/mininet


### Quantization tools (Come soon!)

For fixed-arithmetic quantization of image super resolution.

## Citation
----------
```BibTex
@inproceedings{zhang2021edge,
  title={Edge-oriented Convolution Block for Real-time Super Resolution on Mobile Devices},
  author={Zhang, Xindong and Zeng, Hui and Zhang, Lei},
  booktitle={Proceedings of the 29th ACM International Conference on Multimedia},
  pages={4034--4043},
  year={2021}
}
```


## Acknowledgement
Thanks [EDSR](https://github.com/sanghyun-son/EDSR-PyTorch) for the pioneering work and excellent codebase! The implementation integrated with EDSR is placed on [/legacy](https://github.com/xindongzhang/ECBSR/tree/main/legacy)
