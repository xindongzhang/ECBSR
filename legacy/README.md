# Codes for ECBSR

>[Edge-oriented Convolution Block for Real-time Super Resolution on Mobile Devices](https://www4.comp.polyu.edu.hk/~cslzhang/paper/MM21_ECBSR.pdf) \
> Xindong Zhang, Hui Zeng, Lei Zhang \
> ACM Multimedia 2021


## Codes

This implementation largely depends on [EDSR](https://github.com/sanghyun-son/EDSR-PyTorch). A lighten version will be released soon.

### Dependencies & Installation

The dependencies and installation of code base can refer to [EDSR](https://github.com/sanghyun-son/EDSR-PyTorch). Then, move the archs and network of ECBSR to the code base of EDSR by:

```
mv [ECBSR]/src/option.py [EDSR]/src/
mv [ECBSR]/src/model/ecb.py [EDSR]/src/model
mv [ECBSR]/src/model/ecbsr.py [EDSR]/src/model
```

### Training & Testing
Trained & tested on Pytorch-1.2.0. You could also try less/larger batch-size, if there are limited/enough hardware resources in your GPU-server.
```
cd [EDSR]/src/

## ecbsr-m4c8-x2-prelu(revise the NAME_OF_OUTPUT_FOLDER to your selected folder)
CUDA_VISIBLE_DEVICES=0 python main.py --model ECBSR --scale 2 --patch_size 128 --save NAME_OF_OUTPUT_FOLDER --reset --m_ecbsr 4 --c_ecbsr 8 --ecbsr_idt 0 --act prelu

## ecbsr-m4c8-x4-prelu
CUDA_VISIBLE_DEVICES=0 python main.py --model ECBSR --scale 4 --patch_size 256 --save NAME_OF_OUTPUT_FOLDER --reset --m_ecbsr 4 --c_ecbsr 8 --ecbsr_idt 0 --act prelu

## ecbsr-m4c16-x2-prelu
CUDA_VISIBLE_DEVICES=0 python main.py --model ECBSR --scale 2 --patch_size 128 --save NAME_OF_OUTPUT_FOLDER --reset --m_ecbsr 4 --c_ecbsr 16 --ecbsr_idt 0 --act prelu

## ecbsr-m4c16-x4-prelu
CUDA_VISIBLE_DEVICES=0 python main.py --model ECBSR --scale 4 --patch_size 256 --save NAME_OF_OUTPUT_FOLDER --reset --m_ecbsr 4 --c_ecbsr 16 --ecbsr_idt 0 --act prelu
```

## Hardware deployment

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

## Citation
----------
```BibTex
@article{zhang2021edge,
  title={Edge-oriented Convolution Block for Real-time Super Resolution on Mobile Devices},
  author={Zhang, Xindong and Zeng, Hui and Zhang, Lei},
  booktitle={Proceedings of the 29th ACM International Conference on Multimedia (ACM MM)},
  year={2021}
}
```


## Acknowledgement
This implementation largely depends on [EDSR](https://github.com/sanghyun-son/EDSR-PyTorch). Thanks for the excellent codebase! Our lighten version will come soon.
