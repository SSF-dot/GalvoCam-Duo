### **GalvoCam Duo: a galvanometer-camera stereo system for 3-D reconstruction**



This repository contains the source code and sample datasets for the paper titled "GalvoCam Duo: a galvanometer-camera stereo system for 3-D reconstruction". The code implements the proposed physical model and the calibration procedure. 



#### Environment & Dependencies

To ensure reproducibility, we recommend using **C++ 17** or higher. The core dependencies and their verified versions are listed below: 

| Dependency   | Version | Description                                                  |
| ------------ | ------- | ------------------------------------------------------------ |
| OpenCV       | 4.5.5   | Camera calibration, solve PnP, hand-eye calibration, matrix multiplication |
| Ceres Solver | 1.12.0  | Optimization                                                 |

*Note: The code may work with other versions, but the above configuration has been rigorously tested.* 



#### Dataset Structure

../configuration/camera_parameters.yaml    # Initial camera parameters

../configuration/galvo_parameters.yaml    # Initial galvo parameters

../obj_points/20261029-objcodes.yaml   # Theoretical coordinates of coded arrays on calibration object

../calibration_results/imgCodes.yaml    # Pixel coordinates of coded arrays captured by GalvoCam Duo

../calibration_results/CG.yaml   # Output calibration results



#### Data Processing Pipeline

Directly call the function `void DualCalibration()`.

The main workflow consists of seven stages:

Stage 1: Read the input parameters, including camera-related parameters, galvo-related parameters, 3-D theoretical coordinates of the coded arrays and pixel coordinates of the coded arrays captured by the system.

Stage 2: Calibrate cameras using function `SolveIntrinsic`.

Stage 3: Compute the pose transformations between the virtual cameras and the calibration object using function `SolvePnP`.

Stage 4: Solve the transformations between the camera and the galvo, and between the galvo and the calibration object using function `SolveLeftAXZB` and `SolveRightAXZB`.

Stage 5: Optimize galvo-related parameters and these transformation matrices  using function `Optimize`

Stage 6: Determine the transformation matrix between the two galvos.

Stage 7: Output calibration errors and results.



