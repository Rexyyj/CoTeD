# Datasets
We validated CoTeD with three datasets:
* St. Marc dataset: <https://www.jpjodoin.com/urbantracker/dataset.html> 
* Bird Eye View (BEV) dataset: <https://universe.roboflow.com/mohamed-badreldin-cp2tc/bird-eye-view>
* ImageNet-VidVRD dataset: <https://xdshang.github.io/docs/imagenet-vidvrd.html>

In this folder, we provide a test example for St. Marc dataset. The test data is organized as the structure below,
```bash
.
└── St_Marc_dataset
    ├── coco.names # the object labels
    ├── data
    │   ├── images # the video frames 
    │   └── labels # the labels in each video frames
    └── test_30_fps_long_cleaned.txt # the input video frame sequence to the validation experiments, it is import to organize the video frames in time sequences
```