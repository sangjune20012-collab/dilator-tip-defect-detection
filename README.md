\# Dilator Tip Defect Detection



Research codebase for small defect detection on high-resolution dilator tip inspection images using ROI cropping, patch-based preprocessing, YOLOv11, and RF-DETR.



\## Overview



This repository provides preprocessing, training, evaluation, and visualization scripts for detecting small defects on dilator tip inspection images.



The main pipeline consists of:



1\. Raw dataset split

2\. Dilator tip ROI cropping

3\. Annotation conversion to YOLO format

4\. Patch and sliding-window dataset generation

5\. YOLOv11 training

6\. YOLO-to-COCO conversion

7\. RF-DETR training

8\. Evaluation and paper figure generation



\## Repository Structure



```text

dilator-tip-defect-detection/

├── configs/

├── src/dilator\_tip\_detection/

├── scripts/

├── docs/

├── assets/

├── data/

└── weights/

