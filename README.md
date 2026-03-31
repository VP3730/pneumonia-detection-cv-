Lightweight Pneumonia Detection using Edge Density and HOG Features

📌 Project Overview
This project implements a computer vision-based system to detect pneumonia from chest X-ray images. Instead of relying on heavy deep learning architectures, this approach focuses on computational efficiency by utilizing handcrafted feature extraction combined with a classical machine learning model.

**Core Techniques Used:**
* Edge-based Features: Canny edge detection
* Texture Features: Histogram of Oriented Gradients (HOG)
* Classification: Support Vector Machine (SVM)

 🎯 Problem Statement
The goal of this project is to accurately classify chest X-ray images into one of three distinct categories:
* Class 0: Normal
* Class 1: Pneumonia Type 1
* Class 2:Pneumonia Type 2

 📂 Project Structure
pneumonia-detection-cv/
│
├── dataset/             # (Not included in repo due to size)
│   ├── train_images/
│   ├── test_images/
│   └── labels_train.csv
│
├── src/
│   ├── train.py         # Model training script
│   └── predict.py       # Inference script
│
├── models/              # Saved SVM models will be generated here
├── requirements.txt     # Python dependencies
└── .gitignore

⚙️ Environment Setup
1. Prerequisites
Ensure you have Python installed on your system. You can verify your installation by running:

Bash
python --version
2. Install Dependencies
Clone the repository and install the required Python packages using pip:

Bash
pip install -r requirements.txt

📊 Dataset Setup
Before running the training script, you must download the dataset and structure it exactly as follows within the root directory:

Plaintext
dataset/
├── train_images/
│   └── train_images/
├── test_images/
│   └── test_images/
└── labels_train.csv
Note: The dataset is excluded from this repository due to file size constraints.

🚀 How to Run
1. Train the Model
You must train the model before making any predictions. Run the training script from the root directory:

Bash
python src/train.py
(This will generate and save the model file into the models/ directory).

2. Make Predictions
To test the model on a new image, update the target image path inside src/predict.py, then execute:

Bash
python src/predict.py
Expected Output Format
The script will output the predicted class integer:

Plaintext
Prediction: 0 / 1 / 2

📝 Important Notes
Training First: Always run train.py before attempting to run predict.py so the SVM model is properly generated.

Data Handling: Ensure your image paths match the structure outlined in the Dataset Setup section to prevent FileNotFound errors.


