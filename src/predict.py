import cv2
import numpy as np
import pickle
import os
from skimage.feature import hog
from sklearn.preprocessing import StandardScaler

# === LOAD MODEL ===
model = pickle.load(open("../models/model.pkl", "rb"))

# === IMAGE PATH ===
image_path = r"C:\Users\vandi\OneDrive\Desktop\COLLEGE\8th Sem\Computer Vision\pneumonia-detection\dataset\test_images\test_images\img_91583952582106774.jpg"

print("Path exists:", os.path.exists(image_path))

# === FEATURE FUNCTION ===
def extract_features(image_path):
    img = cv2.imread(image_path)

    if img is None:
        print("Error loading image")
        return None

    img = cv2.resize(img, (128, 128))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    edges = cv2.Canny(gray, 100, 200)
    edge_density = np.sum(edges > 0) / (128 * 128)
    mean_intensity = np.mean(gray)
    edge_variance = np.var(edges)

    hog_features = hog(
        gray,
        pixels_per_cell=(16, 16),
        cells_per_block=(2, 2),
        feature_vector=True
    )

    return np.concatenate((
        [edge_density, mean_intensity, edge_variance],
        hog_features
    ))

# === PREDICT ===
features = extract_features(image_path)

if features is not None:
    features = features.reshape(1, -1)

    # ⚠️ TEMP SCALING (not perfect but works)
    scaler = StandardScaler()
    features = scaler.fit_transform(features)

    prediction = model.predict(features)

    print("Prediction:", prediction[0])
