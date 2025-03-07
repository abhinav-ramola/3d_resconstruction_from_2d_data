import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('TkAgg')  # Change backend to avoid font warnings
import matplotlib.pyplot as plt
from PIL import Image
import tensorflow as tf
from sklearn.model_selection import train_test_split
import tkinter as tk
from tkinter.filedialog import askdirectory, askopenfilename
from tkinter import messagebox
import open3d as o3d

# Set random seed for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# Hyperparameters
IMG_HEIGHT = 250
IMG_WIDTH = 357
BATCH_SIZE = 16
EPOCHS = 10

# Load images and disparity maps from the specified dataset folder structure
def load_data(dataset_dir):
    left_images = []
    right_images = []
    disparity_maps = []

    for folder_name in os.listdir(dataset_dir):
        folder_path = os.path.join(dataset_dir, folder_name)
        if os.path.isdir(folder_path):
            left_image_path = os.path.join(folder_path, 'view1.png')
            right_image_path = os.path.join(folder_path, 'view5.png')
            disparity_map_path_1 = os.path.join(folder_path, 'disp1.png')

            if (os.path.exists(left_image_path) and 
                os.path.exists(right_image_path) and 
                os.path.exists(disparity_map_path_1)):
                
                left_image = Image.open(left_image_path).convert('L').resize((IMG_WIDTH, IMG_HEIGHT))
                right_image = Image.open(right_image_path).convert('L').resize((IMG_WIDTH, IMG_HEIGHT))
                disparity_map_1 = Image.open(disparity_map_path_1).resize((IMG_WIDTH, IMG_HEIGHT))

                left_images.append(np.array(left_image) / 255.0)
                right_images.append(np.array(right_image) / 255.0)
                disparity_maps.append(np.array(disparity_map_1) / 255.0)

    return np.array(left_images), np.array(right_images), np.array(disparity_maps)

def build_model():
    input_left = tf.keras.Input(shape=(IMG_HEIGHT, IMG_WIDTH, 1))
    input_right = tf.keras.Input(shape=(IMG_HEIGHT, IMG_WIDTH, 1))

    conv1 = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same')(input_left)
    conv1 = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same')(conv1)

    conv2 = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same')(conv1)
    conv2 = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same')(conv2)

    merged = tf.keras.layers.Concatenate()([conv2, input_right])

    conv3 = tf.keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same')(merged)
    conv3 = tf.keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same')(conv3)

    output = tf.keras.layers.Conv2D(1, (1, 1), activation='sigmoid')(conv3)

    model = tf.keras.Model(inputs=[input_left, input_right], outputs=output)
    model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])
    
    return model

def create_point_cloud(raw_image_path, disp_matrix, scale_factor=10000):
    disp_matrix = disp_matrix.squeeze()
    height, width = disp_matrix.shape
    img = Image.open(raw_image_path).resize((width, height))
    arr = np.array(img)

    focal_length = 1.0
    baseline = 0.1

    xyzrgb = []
    for y in range(height):
        for x in range(width):
            disparity = disp_matrix[y, x]
            if disparity > 0:
                z = (focal_length * baseline) / disparity
                z *= scale_factor  # Scaling the Z-values for better depth visualization
                xyzrgb.append([x, y, z] + arr[y, x].tolist())

    df = pd.DataFrame(xyzrgb, columns=['x', 'y', 'z', 'r', 'g', 'b'])
    return df


def visualize_disparity(y_true, y_pred):
    y_true_normalized = (y_true.squeeze() - np.min(y_true)) / (np.max(y_true) - np.min(y_true))
    y_pred_normalized = (y_pred - np.min(y_pred)) / (np.max(y_pred) - np.min(y_pred))

    plt.figure(figsize=(12, 6))
    
    plt.subplot(1, 2, 1)
    plt.title("Ground Truth Disparity")
    plt.imshow(y_true_normalized.squeeze(), cmap='gray')
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.title("Predicted Disparity")
    plt.imshow(y_pred_normalized.squeeze(), cmap='gray')
    plt.axis('off')
    
    plt.show()

def visualize_point_cloud(point_cloud_df):
    fig = plt.figure(figsize=(12, 8))
    
    ax = fig.add_subplot(111, projection='3d')
    
    x = point_cloud_df['x']
    y = point_cloud_df['y']
    z = point_cloud_df['z']
    
    r = point_cloud_df['r'] / 255.0
    g = point_cloud_df['g'] / 255.0
    b = point_cloud_df['b'] / 255.0
    
    ax.scatter(x, y, z, c=np.stack([r, g, b], axis=1), s=0.5) # Adjusted size for better visualization
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    
    plt.title("3D Point Cloud Visualization")
    
    plt.show()

def load_and_process_point_cloud(file_path):
   data = np.genfromtxt(file_path, delimiter=',', skip_header=1)
   
   # Extracting coordinates and colors
   x = data[:, 0].astype(float)
   y = data[:, 1].astype(float)
   z = data[:, 2].astype(float)  
   colors = data[:, 3:6].astype(int)

   # Dynamically scale Z-values based on the min and max Z-values 
   z_min_max_values_range_adjusted_for_visualization(z)

   pcd=o3d.geometry.PointCloud()
   pcd.points=o3d.utility.Vector3dVector(np.column_stack((x,y,z)))
   pcd.colors=o3d.utility.Vector3dVector(colors/255.0)

   print(f"Point Cloud has {len(pcd.points)} points")
   print(f"First few points: {np.asarray(pcd.points)[:5]}")
   print(f"First few colors: {np.asarray(pcd.colors)[:5]}")

   return pcd

def z_min_max_values_range_adjusted_for_visualization(z_values):
    min_z = np.min(z_values)
    max_z = np.max(z_values)
    
    # Normalize Z-values to be between 0 and 1
    z_scaled = (z_values - min_z) / (max_z - min_z)
    
    return z_scaled

def create_mesh_from_point_cloud(pcd):
   # Estimate normals with improved parameters for better quality mesh generation 
   pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.05,max_nn=30))
   
   # Perform Poisson surface reconstruction with increased depth for finer detail 
   mesh,densities=o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd ,depth=10)
   
   # Simplify and smooth the mesh with adjusted parameters for better quality 
   mesh=mesh.simplify_quadric_decimation(target_number_of_triangles=100000) 
   mesh=mesh.filter_smooth_simple(number_of_iterations=15)

   # Visualize the point cloud and mesh 
   o3d.visualization.draw_geometries([pcd ,mesh])

def main():
    root = tk.Tk()
    root.withdraw()

    dataset_dir = askdirectory(title="Select Dataset Directory")
    if not dataset_dir:
        messagebox.showerror("Error", "Dataset directory not selected!")
        return

    left_images, right_images, disparities = load_data(dataset_dir)
    if len(disparities) == 0:
        messagebox.showerror("Error", "No disparities found in the dataset!")
        return

    X_left_train, X_left_test, X_right_train, X_right_test, y_train, y_test = train_test_split(
        left_images.reshape(-1, IMG_HEIGHT, IMG_WIDTH, 1),
        right_images.reshape(-1, IMG_HEIGHT, IMG_WIDTH, 1),
        disparities.reshape(-1, IMG_HEIGHT, IMG_WIDTH),
        test_size=0.2,
        random_state=42
    )

    # Check if the model already exists
    model_path = 'stereo_model.h5'
    if os.path.exists(model_path):
        print("Loading existing model...")
        model = tf.keras.models.load_model(model_path)
    else:
        print("Training new model...")
        model = build_model()
        model.fit(
            [X_left_train, X_right_train],
            y_train,
            batch_size=BATCH_SIZE,
            epochs=EPOCHS,
            validation_split=0.1,
            verbose=1
        )

        # Save the trained model
        model.save(model_path)
        print(f"Model saved to {model_path}")

    test_loss, test_mae = model.evaluate([X_left_test, X_right_test], y_test)
    print(f"Test Loss: {test_loss}, Test MAE: {test_mae}")

    index_to_predict = int(input(f"Enter index of image to predict (0 to {len(X_left_test)-1}): "))

    if index_to_predict < len(X_left_test):
        predicted_disparity_map = model.predict([X_left_test[index_to_predict:index_to_predict+1],
                                                 X_right_test[index_to_predict:index_to_predict+1]])[0]

        visualize_disparity(y_test[index_to_predict:index_to_predict+1], predicted_disparity_map)

        raw_image_path_left = askopenfilename(title="Select Left Image for Point Cloud")
        
        if not raw_image_path_left:
            messagebox.showerror("Error", "Left image for point cloud not selected!")
            return

        point_cloud_df = create_point_cloud(raw_image_path=raw_image_path_left, disp_matrix=predicted_disparity_map)

        output_dir = './data/output'
        os.makedirs(output_dir, exist_ok=True)

        output_txt_path = os.path.join(output_dir, f'point_cloud_{index_to_predict}.csv')

        point_cloud_df.to_csv(output_txt_path, index=False)

        print(f"Point cloud saved to {output_txt_path}")

        visualize_point_cloud(point_cloud_df)

        pcd = load_and_process_point_cloud(output_txt_path)

        create_mesh_from_point_cloud(pcd)


if __name__=='__main__':
   main()
