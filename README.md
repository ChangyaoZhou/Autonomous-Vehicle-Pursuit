#  Autonomous-Vehicle-Pursuit
## Robust Autonomous Vehicle Pursuit without Supervised Steering Labels 

 This repository contains code for the paper **Robust Autonomous Vehicle Pursuit without Supervised Steering Labels** submitted to IEEE Robotic Automation Letter. 

![image](./images/pipeline_new.png)

In this work, we present a learning method for lateral and longitudinal motion control of an ego-vehicle for the task of vehicle pursuit. 
To train our model, we do not rely on steering labels recorded from an expert driver, but effectively leverage classical controller as an offline label generation tool.
In addition, we account for the errors in the predicted control values, which can lead to crashes of the controlled vehicle. 
To this end, we propose an effective geometry-based data augmentation approach, which allows to train a network that is capable of handling different views of the target vehicle. 

During the pursuit,  target pose of the followed vehicle with respect to the ego-vehicle is firstly estimated using a Convolutional Neural Network. 
This information is then fed to a Multi-Layer Perceptron, which regresses the control commands for the ego-vehicle, namely throttle and steering angle. 
We extensively validate our approach using the CARLA simulator on a wide range of terrains and weather conditions. 
Our method demonstrates real-time performance, robustness to different scenarios including unseen trajectories and high route completion.

## Results

Here we show videos to show the qualitative results of our approach **in different maps**, in a city (left) and in the countryside (right). 
To summarize, the first vehicle (red) in the video is controlled by autopilot. The ego-vehicle (black) is controlled autonomously with our method to follow the red target vehicle.
<div align="center"><img src=./images/1ego_cat.gif width=" 500 "></div>  

We further tested our model with **two ego vehicles**. As can be seen in the ego vehicle, the first vehicle (red) is controlled by autopilot, the first ego-vehicle (gray) is controlled by our model to follow the red target vehicle. And the second ego-vehicle (black) is controlled by the same model to follow the gray ego-vehicle.
<div align="center"><img src=./images/2ego_cat.gif width=" 500 "></div>  

Though the model is trained only with samples in sunny weather, it fits **different weather conditions** quite well. Here we show two examples, in dark night (left) and rainy weather (right).
<div align="center"><img src=./images/weather_cat.gif width=" 500 "></div>  


## Environment

Clone the repo, setup CARLA 0.9.11, and build the conda environment:

```
conda create -n myenv python=3.7 
conda activate myenv
conda install --file requirements.txt
```

For installation of pytorch3d, you can refer to [Pytorch3d - Tutorial](https://pytorch3d.org/tutorials/bundle_adjustment)

For installation of CARLA 0.9.11, you can refer to [CARLA 0.9.11](https://github.com/carla-simulator/carla#building-carla)

### Licences
**CARLA licenses**

CARLA specific code is distributed under MIT License.
CARLA specific assets are distributed under CC-BY License.

## Data Collection
## Data Augmentation
## Training
## Inference and Evaluation


## Pretrained models:

| Model | CNN/MLP | Path|
| :-----:| :----: | :----: |
| Baseline | CNN | models/pretained_models/Baseline.pth |
| Three-camera | CNN | models/pretained_models/Three-camera.pth |
| SS depth + 3D detector (our approach) | CNN | models/pretained_models/SS_depth+3D_detector.pth |
| Stereo depth + 3D detector (our approach) | CNN | models/pretained_models/Stereo_depth+3D_Detector.pth |
| ground truth depth + 3D detector | CNN | models/pretained_models/GT_depth+3D_detector.pth |
| SS depth + ground truth transformation | CNN | models/pretained_models/SS_depth+GT_transformation.pth |
| Stereo depth +  ground truth transformation | CNN | models/pretained_models/Oracle.pth |
| Oracle | CNN | models/pretained_models/Stereo_depth+3D_Detector.pth |
| Random Noise Injection | CNN | models/pretained_models/Random_Noise_Injection.pth |
| MLP | MLP | models/pretained_models/MLP.pth |
> 3470ca8525337d556546dd898105070334c50dd9


