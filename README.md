# Scene Understanding for Robotic Navigation

This project focuses on developing a robotic system capable of understanding its surroundings, detecting and segmenting objects, and navigating to a specified target object within a known environment.

## Installation

To get started, install the required dependencies:

```bash
pip install -r requirements.txt
Object Detection using YOLO
We use YOLO (You Only Look Once) for real-time object detection. The implementation is available in the yolo/yolo_detection.py file.

Object Segmentation using Mask R-CNN
For object segmentation, we utilize Mask R-CNN. The implementation is available in the mask_rcnn/mask_rcnn_segmentation.py file.

SLAM for Scene Mapping and Localization
We use SLAM (Simultaneous Localization and Mapping) to create a map of the environment and localize the robot. Detailed resources and setup instructions are available in the slam/slam_integration.md file.

Path Planning using A*
The A* algorithm is used for path planning. The implementation is available in the path_planning/a_star.py file.

Integration
The integration of object detection, segmentation, SLAM, and path planning is described in the integration/integrate_all.py file.

