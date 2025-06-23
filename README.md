# Voice_Clustering
This course project focuses on predicting the gender of a speaker (male/female) using a voice dataset with 46 extracted audio features. By leveraging ensemble learning techniques like Random Forest and a custom Power Boosting method, the model effectively classifies voices with improved accuracy and balanced distribution.

# Objective
To build a machine learning model that classifies voice samples as either male or female based on 46 acoustic features derived from audio recordings.

# Technologies & Tools
Python
Streamlit (for UI and deployment)
Random Forest Classifier (scikit-learn)
Power Boosting (custom linear distribution technique to improve balance)
SMOTE
Pandas, NumPy, Matplotlib, Seaborn

# Workflow
Data Cleaning & Preprocessing
Exploratory Data Analysis (EDA)
Model Building
Trained a Random Forest model
Implemented Power Boosting for better class distribution
Web Deployment
Built a user-friendly interface using Streamlit
Allows users to input values for the 46 features and get real-time predictions

# Dataset
16148 voice samples
46 extracted features (e.g., meanfreq, IQR, centroid, etc.)
Target label: male or female

# Project Structure
bash
Copy
Edit
voice-gender-classification/
├── Voice_gender_detection_app.py      # Streamlit app
├── vocal_gender_features_new/         # CSV dataset
├── rfcl.pkl/                          # Saved model (Pickle/Joblib)                  
├── human_voice_clustering/            # EDA & training experiments
└── README.md
