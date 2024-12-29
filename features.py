import cv2
import numpy as np
import os
from skimage.feature import hog
from skimage.feature import local_binary_pattern


def extract_grid_color_features(image, grid_size=3):
    h, w, _ = image.shape
    grid_h, grid_w = h // grid_size, w // grid_size
    features = []
    for i in range(grid_size):
        for j in range(grid_size):
            cell = image[i*grid_h:(i+1)*grid_h, j*grid_w:(j+1)*grid_w]
            hist = cv2.calcHist([cell], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
            features.append(cv2.normalize(hist, hist).flatten())
    return np.concatenate(features)


def extract_lbp_features(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    lbp = local_binary_pattern(gray, P=8, R=1, method='uniform')
    hist, _ = np.histogram(lbp.ravel(), bins=np.arange(0, 11), range=(0, 10))
    return hist / hist.sum()  # Normalize histogram


def extract_hog_features(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    hog_features, _ = hog(gray, orientations=9, pixels_per_cell=(8, 8), 
                          cells_per_block=(2, 2), visualize=True)
    return hog_features

def extract_features(image_path):
    # Load the image
    image = cv2.imread(image_path)
    image = cv2.resize(image, (128, 128))  # Resize for consistency
    
    # 1. Color Histogram (Global)
    hist = cv2.calcHist([image], [0, 1, 2], None, [32, 32, 32], [0, 256, 0, 256, 0, 256])
    hist = cv2.normalize(hist, hist).flatten()
    
    # 2. Edge Detection
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 100, 200)
    edges_flat = edges.flatten() / 255.0  # Normalize edges to [0, 1]
    
    # 3. Spatial Color Features
    grid_color_features = extract_grid_color_features(image, grid_size=3)
    
    # 4. Texture Features (LBP)
    lbp_features = extract_lbp_features(image)
    
    # 5. Shape Features (HOG)
    hog_features = extract_hog_features(image)
    
    # Combine all features
    features = np.concatenate([hist, edges_flat, grid_color_features, lbp_features, hog_features])
    return features


def process_dataset(dataset_path):
    data = []
    labels = []
    count = -1
    for class_folder in os.listdir(dataset_path):
        count = count + 1
        if count == 5:
            break
        print("Class: ", class_folder)
        class_path = os.path.join(dataset_path, class_folder)
        if os.path.isdir(class_path):
            for image_file in os.listdir(class_path):
                image_path = os.path.join(class_path, image_file)
                features = extract_features(image_path)
                data.append(features)
                labels.append(class_folder)
    return np.array(data), np.array(labels)

# Extract features for the dataset
data, labels = process_dataset("dataset")
np.save("features.npy", data)
np.save("labels.npy", labels)
