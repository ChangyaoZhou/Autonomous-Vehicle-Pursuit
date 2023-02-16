# Autonomous-Vehicle-Pursuit
< HEAD

Repository for Autonomous Vehicle Pursuit including video, code coming soon...

Here we share the video to show the qualitative results of our approach for car following. To evaluate the potential of our method in the context of car platooning, we added a second ego-vehicle to follow the first ego-vehicle.

To summarize, the first vehicle (red) in the video is controlled by autopilot and has been referred to as the target in this work. The ego-vehicle (in gray) is controlled autonomously using our method to follow the red target vehicle. The second ego-vehicle (in black) is also controlled autonomously by our method but follows its predecessor, i.e. the first ego-vehicle (gray).

Here we share the video to show the qualitative results of our approach for car following. To evaluate the potential of our method in the context of car platooning, we added a second ego-vehicle to follow the first ego-vehicle.

To summarize, the first vehicle (red) in the video is controlled by autopilot and has been referred to as the target in this work. The ego-vehicle (in gray) is controlled autonomously using our method to follow the red target vehicle. The second ego-vehicle (in black) is also controlled autonomously by our method but follows its predecessor, i.e. the first ego-vehicle (gray).

Pretrained models:

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


