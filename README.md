Image Super-Resolution Using Convolutional Neural Networks

This is an implementation of [SRCNN](https://arxiv.org/abs/1501.00092) using PyTorch and OpenCV.

![alt text](https://pub.mdpi-res.com/applsci/applsci-10-00854/article_deploy/html/images/applsci-10-00854-g001.png?1582186393)

-----

## Download Datasets

Contains DIV2K, DIV8K, Flickr2K, OST, T91, Set5, Set14, BSDS100 and BSDS200, etc.

- [Data](https://drive.google.com/drive/folders/1A6lzGeQrFMxPqJehK9s37ce-tPDj20mD?usp=sharing)


## Preprocessing

Each dataset is saved in a h5 file with a specific upscale factor. I use the OpenCV library - [cv2.INTER_CUBIC] interpolation instead of PIL library.

To run we need to specify which dataset we want to use, patch size, stride, upscale factor etc.

Example

```
./preprocessing.py --dataset=T91 --output-path=T91
                   --patch-size=33 --stride=14 --upscale-factor=3
                   --validation=False
```
