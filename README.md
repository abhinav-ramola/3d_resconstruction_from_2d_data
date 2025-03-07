# 3D Image Reconstruction from 2D Data

## Overview
This project focuses on reconstructing 3D images from 2D stereo image pairs using a deep learning model. The model predicts disparity maps, which are then used to generate a 3D point cloud.

## Features
- Load stereo image pairs from a dataset
- Train or load a CNN-based model to predict disparity maps
- Visualize ground truth and predicted disparity maps
- Generate and visualize a 3D point cloud from the predicted disparity
- Mesh reconstruction from the point cloud

## Dependencies
Ensure you have the following libraries installed before running the project:

```bash
pip install numpy pandas matplotlib pillow tensorflow scikit-learn open3d tkinter
```

## Dataset Structure
The dataset should be organized as follows:
```
/dataset_dir/
    /sample1/
        view1.png  # Left image
        view5.png  # Right image
        disp1.png  # Ground truth disparity
    /sample2/
        view1.png
        view5.png
        disp1.png
```

## Running the Project
### 1. Load and Process Data
Select the dataset directory using the GUI prompt.

### 2. Train or Load Model
- If a trained model (`stereo_model.h5`) exists, it will be loaded.
- Otherwise, the model will be trained using the dataset.

### 3. Predict Disparity Map
- Enter an index to predict disparity for a test image.
- The predicted disparity will be visualized.

## 4. Generate and Visualize 3D Point Cloud
- Select a left image for point cloud generation.
- The generated point cloud will be saved as a CSV file.
- The point cloud can be visualized in a 3D plot.

### Example:
**Input 2D Image → Generated 3D Point Cloud**
<p align="center">
  <img src="https://raw.githubusercontent.com/abhinav-ramola/3d_resconstruction_from_2d_data/main/view1.png" alt="Input 2D Image" width="300">
  <img src="https://raw.githubusercontent.com/abhinav-ramola/3d_resconstruction_from_2d_data/main/Point_Cloud.png" alt="Generated Point Cloud" width="300">
</p>

## 5. Mesh Reconstruction
- The point cloud is processed to generate a mesh using Open3D.
- The reconstructed mesh can be visualized.

### Example:
**Generated 3D Model**
<p align="center">
  <img src="https://raw.githubusercontent.com/abhinav-ramola/3d_resconstruction_from_2d_data/main/3d.png" alt="3D Model Output" width="400">
</p>


## Output Files
- `stereo_model.h5`: Trained model
- `./data/output/point_cloud_<index>.csv`: Generated point cloud

## Notes
- The `scale_factor` parameter is used to adjust depth visualization.
- The project uses `Tkinter` for file selection and GUI prompts.
- The mesh reconstruction uses Poisson surface reconstruction.

## Future Improvements
- Optimize the CNN model for better accuracy.
- Implement a more advanced 3D reconstruction approach.
- Improve real-time performance for disparity prediction.

## Authors
Developed as part of a mini-project on 3D image reconstruction using deep learning and computer vision techniques.


