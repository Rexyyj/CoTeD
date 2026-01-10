# CoTeD
CoTeD targets UAV–edge cooperative video object detection scenarios where a DNN is split between a resource-constrained UAV and a edge server over a private 5G network. It proposes a goal-oriented framework (as the figure below shows) that, **without model retraining**, can dynamically prunes, compresses, and reconstructs DNN activation tensors explicitly trading off radio bandwidth consumption against inference accuracy and latency in real time.
![](/figures/coted.png)

This repository contains the implementation and descriptions of the CoTeD framework and testbed presented in our publication:
>Yenchia Yu, Matteo Mendula, Marco Levorato, Marina Papatriantafilou, and Carla Fabiana Chiasserini. Efficient Tensor Compression and Reconstruction for Edge-Based Object Detection. {Publication info to be added}

## How to use this Repo

### 1. Testbed setup
CoTeD is designed and validated on a UAV-5G-Edge testbed.
For the testbed architecture and setup, please see the description in [```./testbed/```](https://github.com/Rexyyj/CoTeD/tree/master/testbed).

### 2. Running experiment

