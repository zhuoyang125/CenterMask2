## CenterMask with Open Images

This repo trains CenterMask (built on Detectron2) on Open Images datasets. Credits to Youngwan Lee and Jongyoul Park arxiv: https://arxiv.org/abs/1911.06667 CenterMask original repo: https://github.com/youngwanLEE/centermask2

### Hyperparameter Testing

Comparing 3 sets of training hyperparameters: 
- gradient clip value 0.5, lr 0.05, lr decay 0.2 x 2 (blue) 
- gradient clip value 0.5, lr 0.005 (orange)
- lr 0.002, lr decay 0.2 x 2 (red)


<h3>total loss</h3>
<img src="https://user-images.githubusercontent.com/48168603/80370669-7711d600-8845-11ea-87d0-19dd5365b6ec.jpg" width=80%/>
<!-- <img src="https://drive.google.com/uc?export=view&id=1ueTOzUnrGCrc6bQ0TpCeu9Zbbw4UOjtj" width=40%/>
<img src="https://drive.google.com/uc?export=view&id=1gwqUv7svQyR9b5td0tWprD4Jpl8gs9dL" width=40%/>
<img src="https://drive.google.com/uc?export=view&id=1Uzu0cSaYNekiy894rZwq6R5EclUPupMJ" width=40%/>
<img src="https://drive.google.com/uc?export=view&id=1Uzu0cSaYNekiy894rZwq6R5EclUPupMJ" width=40%/>
<img src="https://drive.google.com/uc?export=view&id=1TW4FK5mhvRVvMUBG_zXGJ1ryUXMbZuzA" width=40%/> -->

### Open Images Training and Results

Mask RCNN and CenterMask trained on Open Images V6, containing 24,591 images and 57,812 masks of 300 classes. Due to imbalanced distribution amongst classes, repeat factor sampling was used to oversample tail classes, learning rate schedule x2.

*All results measured on NVIDIA Quadro P1000 

|Method|Backbone|Inference Time(s)|mask AP|box AP|full metrics|
|:-----:|:----:|:-------:|:--:|:--:|:---:|
|CenterMask|VoVNetV2-19|0.16|15.513|14.594|[metrics](https://drive.google.com/file/d/1esmLMGEiaRPW4XYz31EjxqGDIuQO2YDG/view?usp=sharing)
|Mask-RCNN|ResNet-50|0.48|17.765|15.512|[metrics](https://drive.google.com/file/d/17hxrIMGy0Z8N-xrZKqLNr87MjHIp5Ct8/view?usp=sharing)
|CenterMask|SimpleNet|0.56|7.944|7.073|[metrics](https://drive.google.com/file/d/1N0qXBcV0PLj6YXUYABL3Co-jDoLOUbp0/view?usp=sharing)
|CenterMask|EfficientNet-B0|0.29|3.753|3.345|[metrics](https://drive.google.com/file/d/1wa4Sd-2XVZFeWEy0uIrjNXIRRKlCBwTO/view?usp=sharing)

### Side-by-Side Comparison

#### CenterMask-VoVNet
<img src="https://user-images.githubusercontent.com/48168603/83113962-2c21f300-a07d-11ea-8064-8ece48094902.gif" width=60%/>

#### Mask RCNN
<img src="https://user-images.githubusercontent.com/48168603/83113916-1e6c6d80-a07d-11ea-9a71-47943c7bfd87.gif" width=60%/>

#### CenterMask-SimpleNet
<img src="https://user-images.githubusercontent.com/48168603/83113996-3643f180-a07d-11ea-8f1d-16ce986c7a6c.gif" width=60%/>

#### CenterMask-EfficientNetB0
<img src="https://user-images.githubusercontent.com/48168603/83113946-262c1200-a07d-11ea-9fc7-efa312f9aa00.gif" width=60%/>

### Image or Video Demo

To run inference on images or video, run `CenterMask2/custom_demo.py` for Mask RCNN or `CenterMask2/projects/CenterMask2/custom_demo.py` for CenterMask. Run it with:
```
python custom_demo.py --config_file configs/COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml --input input1.jpg input2.jpg --output results/ --confidence_threshold 0.6 --weights model_final.pth
```

For video inference, replace --input files with --video video.mp4, and --output is name of output video saved in current dir. 

If no --output given, instances are shown in cv2 window

### Training on Open Images

Dataset has to be loaded as a json file in COCO format, together with a folder of training images. Every mask must have an annotation, with at least 6 polygon points in 'segmentation'
```
annotation{
"id": int, 
"image_id": int,
"category_id": int, 
"segmentation": RLE or [polygon], 
"area": float, 
"bbox": [x,y,width,height], 
"iscrowd": 0 or 1
}

categories[{
"id": int, 
"name": str, 
"supercategory": str
}]
```

Change the json file and training image directory for `get_train_dicts()` in `openimages_utils/data_dicts.py` to that of dataset you want to train. 

To train, edit `cfg.merge_from_file('path/to/config/file')` in `train.py` with config file of choice. Load model weights at `cfg.MODEL.WEIGHTS = 'path/to/model/weights'`. Then, simply execute `train.py`

Training with EfficientNet backbone requires installation of Pytorch EfficientNet from (https://github.com/lukemelas/EfficientNet-PyTorch), files added to 'centermask/modeling/backbone' folder. The pretrained model is loaded in `efficientnet.py` 

### Validation

Change json file and validation image directory for `get_val_dicts()` in `openimages_utils/data_dicts.py` to that validation dataset.

Run it with
```
python validate.py --configs configs/COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml --model_pth model_final.pth --mode evaluate --threshold 0.5
```
where --mode is either 'infer' or 'evaluate' on validation images.

***

### Requirements
- Python >= 3.6(Conda)
- PyTorch 1.3
- [torchvision](https://github.com/pytorch/vision/) that matches the PyTorch installation.
	You can install them together at [pytorch.org](https://pytorch.org) to make sure of this.
- OpenCV, needed by demo and visualization
- [fvcore](https://github.com/facebookresearch/fvcore/): `pip install git+https://github.com/facebookresearch/fvcore`
- pycocotools: `pip install cython; pip install git+https://github.com/philferriere/cocoapi.git#subdirectory=PythonAPI`
- VS2019(no test in older version)/CUDA10.1(no test in older version)

### several files must be changed by manually.
```
file1: 
  {your evn path}\Lib\site-packages\torch\include\torch\csrc\jit\argument_spec.h
  example:
  {C:\Miniconda3\envs\py36}\Lib\site-packages\torch\include\torch\csrc\jit\argument_spec.h(190)
    static constexpr size_t DEPTH_LIMIT = 128;
      change to -->
    static const size_t DEPTH_LIMIT = 128;
file2: 
  {your evn path}\Lib\site-packages\torch\include\pybind11\cast.h
  example:
  {C:\Miniconda3\envs\py36}\Lib\site-packages\torch\include\pybind11\cast.h(1449)
    explicit operator type&() { return *(this->value); }
      change to -->
    explicit operator type&() { return *((type*)this->value); }
```

### Build detectron2

After having the above dependencies, run:
```
conda activate {your env}

"C:\Program Files (x86)\Microsoft Visual Studio\2019\Enterprise\VC\Auxiliary\Build\vcvars64.bat"

git clone https://github.com/conansherry/detectron2

cd detectron2

python setup.py build develop
```
Note: you may need to rebuild detectron2 after reinstalling a different build of PyTorch.

<div align="center">
  <img src="docs/windows_build.png"/>
</div>

<img src=".github/Detectron2-Logo-Horz.svg" width="300" >

Detectron2 is Facebook AI Research's next generation software system
that implements state-of-the-art object detection algorithms.
It is a ground-up rewrite of the previous version,
[Detectron](https://github.com/facebookresearch/Detectron/),
and it originates from [maskrcnn-benchmark](https://github.com/facebookresearch/maskrcnn-benchmark/).

### What's New
* It is powered by the [PyTorch](https://pytorch.org) deep learning framework.
* Includes more features such as panoptic segmentation, densepose, Cascade R-CNN, rotated bounding boxes, etc.
* Can be used as a library to support [different projects](projects/) on top of it.
  We'll open source more research projects in this way.
* It [trains much faster](https://detectron2.readthedocs.io/notes/benchmarks.html).

See our [blog post](https://ai.facebook.com/blog/-detectron2-a-pytorch-based-modular-object-detection-library-/)
to see more demos and learn about detectron2.

## Installation

See [INSTALL.md](INSTALL.md).

## Quick Start

See [GETTING_STARTED.md](GETTING_STARTED.md),
or the [Colab Notebook](https://colab.research.google.com/drive/16jcaJoc6bCFAQ96jDe2HwtXj7BMD_-m5).

Learn more at our [documentation](https://detectron2.readthedocs.org).
And see [projects/](projects/) for some projects that are built on top of detectron2.

## Model Zoo and Baselines

We provide a large set of baseline results and trained models available for download in the [Detectron2 Model Zoo](MODEL_ZOO.md).


## License

Detectron2 is released under the [Apache 2.0 license](LICENSE).

## Citing Detectron

If you use Detectron2 in your research or wish to refer to the baseline results published in the [Model Zoo](MODEL_ZOO.md), please use the following BibTeX entry.

```BibTeX
@misc{wu2019detectron2,
  author =       {Yuxin Wu and Alexander Kirillov and Francisco Massa and
                  Wan-Yen Lo and Ross Girshick},
  title =        {Detectron2},
  howpublished = {\url{https://github.com/facebookresearch/detectron2}},
  year =         {2019}
}
```
