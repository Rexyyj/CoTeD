# CoTeD
CoTeD targets UAV–edge cooperative video object detection scenarios where a DNN is split between a resource-constrained UAV and a edge server over a private 5G network. It proposes a goal-oriented framework (as the figure below shows) that, **without model retraining**, can dynamically prunes, compresses, and reconstructs DNN activation tensors explicitly trading off radio bandwidth consumption against inference accuracy and latency in real time.
![](/figures/coted.png)

This repository contains the implementation and descriptions of the CoTeD framework and testbed presented in our publication:
>Yenchia Yu, Matteo Mendula, Marco Levorato, Marina Papatriantafilou, and Carla Fabiana Chiasserini. Efficient Tensor Compression and Reconstruction for Edge-Based Object Detection. {Publication info to be added}

## How to use this Repository

### Testbed setup
CoTeD is designed and validated on a UAV-5G-Edge testbed.
For the testbed architecture and setup, please see the description in [```./testbed/```](https://github.com/Rexyyj/CoTeD/tree/master/testbed).

### Running experiment
In this repository, we provide the example codes to run the experiment with YOLOv3-tiny model (under folder ```./coted_framework/yolov3_tiny_experiment/```) and YOLOv3 model (under folder ```./coted_framework/yolov3_experiment/```). 

We provide the find-tuned YOLOv3-tiny model checkpoints in folder ```./coted_framework/ckpt/```. For testing with other datasets, please check the description in ```./pytorchyolo/``` to perform fine-tuning. 

Running the validation experiment of CoTeD framework with the following steps (use YOLOv3-tiny model as example),

1 . Update the edge server IP address and port according to the testbed setup:
```python
# In ./coted_framework/yolov3_tiny_experiment/experiment_edge.py
...
cherrypy.server.socket_host = "10.0.1.34"
cherrypy.config.update({'server.socket_port':8092 })
...

# In ./coted_framework/yolov3_tiny_experiment/experiment_client.py
...
service_uri = "http://10.0.1.34:8092/tensor"
reset_uri = "http://10.0.1.34:8092/reset"
...
```

2. Start the CoTeD edge and DNN tail at the edge server side with, 
```bash
python3 experiment_edge.py <DNN_split_layer>
# For YOLOv3-tiny, we split at the 8th layer. For YOLOv3 model, we split at the 10th layer.
```

3. Run the CoTeD mobile and DNN head at the UAV computation unit (i.e., the Nvidia Jetson Orin Nano) with, 
```bash
python3 experiment_client.py <DNN_split_layer>```
# For YOLOv3-tiny, we split at the 8th layer. For YOLOv3 model, we split at the 10th layer.
```

4. Find the experiment measurements under folder ```./coted_framework/measurements/```.


### Test with different datasets
To test the CoTeD framework with different datasets, the following configurations should be updated before running the experiment (use YOLOv3-tiny model as example),

1. Fine-tuned YOLOv3-tiny model checkpoints:
```python
# In ./coted_framework/yolov3_tiny_experiment/experiment_edge.py
cfg_path = "../configs/yolov3_tiny.cfg"
model_path = "../ckpt/stmarc_tiny.pth" # to the fine-tuned model checkpoint

# In ./coted_framework/yolov3_tiny_experiment/experiment_client.py
cfg_path = "../configs/yolov3_tiny.cfg"
model_path = "../ckpt/stmarc_tiny.pth" # to the fine-tuned model checkpoint
```

2. The SNR to mAP/Sensitivity drop curve parameters:
```python
# In ./coted_framework/manager/manager_full.py

# STMARC [k, h, b]
self.map_curve = [0.062, 0.917, 0]
self.sen_curve = [0.067, 0.654, 2]
```

3. The test video frame sequences and label names:
```python
# In ./coted_framework/yolov3_tiny_experiment/experiment_client.py
testdata_path = "../../datasets/St_Marc_dataset/test_30_fps_long_cleaned.txt"
class_name_path = "../../datasets/St_Marc_dataset/coco.names"
```