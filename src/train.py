print("STARTING SCRIPT...")

import os
import cv2
import numpy as np
import pandas as pd
import pickle

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler

from skimage.feature import hog

# === PATHS ===
TRAIN_FOLDER = r"C:\Users\vandi\OneDrive\Desktop\COLLEGE\8th Sem\Computer Vision\pneumonia-detection\dataset\train_images\train_images"
LABELS_PATH = r"C:\Users\vandi\OneDrive\Desktop\COLLEGE\8th Sem\Computer Vision\pneumonia-detection\dataset\labels_train.csv"

# === FEATURE FUNCTION ===
def extract_features(image_path):
    try:
        img = cv2.imread(image_path)

        if img is None:
            return None

        img = cv2.resize(img, (128, 128))
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # --- EDGE FEATURES (YOUR CORE IDEA) ---
        edges = cv2.Canny(gray, 100, 200)
        edge_density = np.sum(edges > 0) / (128 * 128)
        mean_intensity = np.mean(gray)
        edge_variance = np.var(edges)

        # --- HOG FEATURES (ADDED BOOST) ---
        hog_features = hog(
            gray,
            pixels_per_cell=(16, 16),
            cells_per_block=(2, 2),
            feature_vector=True
        )

        # combine both
        return np.concatenate((
            [edge_density, mean_intensity, edge_variance],
            hog_features
        ))

    except:
        return None


# === LOAD CSV ===
df = pd.read_csv(LABELS_PATH)

X = []
y = []

print("Processing images...")

files_in_folder = os.listdir(TRAIN_FOLDER)
print("Total files in folder:", len(files_in_folder))

# === MATCH FILES ===
for index, row in df.iterrows():
    csv_name = str(row["file_name"]).strip()
    label = int(row["class_id"])

    matched_file = None

    for f in files_in_folder:
        if csv_name in f or f in csv_name:
            matched_file = f
            break

    if matched_file is None:
        continue

    img_path = os.path.join(TRAIN_FOLDER, matched_file)

    features = extract_features(img_path)

    if features is not None:
        X.append(features)
        y.append(label)

# === CHECK DATA ===
print("Total valid images:", len(X))

if len(X) == 0:
    print("❌ ERROR: No valid images processed.")
    exit()

X = np.array(X)
y = np.array(y)

# === FEATURE SCALING (CRITICAL BOOST) ===
scaler = StandardScaler()
X = scaler.fit_transform(X)

# === SPLIT DATA ===
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("Training model...")

# === MODEL ===
model = SVC(kernel='rbf', C=10, gamma='scale')
model.fit(X_train, y_train)

# === EVALUATE ===
y_pred = model.predict(X_val)
accuracy = accuracy_score(y_val, y_pred)

print("✅ Validation Accuracy:", accuracy)

# === SAVE MODEL ===
os.makedirs("../models", exist_ok=True)

with open("../models/model.pkl", "wb") as f:
    pickle.dump(model, f)

print("✅ Model trained and saved!")
