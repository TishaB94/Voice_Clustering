import streamlit as st
import pandas as pd
import numpy as np
import joblib  # for loading saved model

# Load your trained model (make sure you have saved it as 'model.pkl')
rfcl = joblib.load('rfcl.pkl')

st.success("Model loaded successfully!")

try:
    rfcl = joblib.load('rfcl.pkl')
    st.success("Model loaded successfully!")
    
    # Debug: Show model's expected features
    if hasattr(rfcl, 'feature_names_in_'):
        st.write(f"Model expects {rfcl.n_features_in_} features:", rfcl.feature_names_in_)
    else:
        st.warning("Model doesn't have feature names. Using hardcoded feature list.")
        
except Exception as e:
    st.error(f"Error loading model: {str(e)}")
    st.stop()




st.title("Voice Gender Classification")

st.write("""
This app predicts whether a voice is Male (1) or Female (0) based on audio features.
""")
       

# Example: Input features from user (customize with your important features)


mean_spectral_centroid	= st.number_input('mean_spectral_bandwidth', value=2226.033)
std_spectral_centroid = st.number_input('Std Spectral Centroid', value= 1462.589)
mean_spectral_bandwidth = st.number_input('Mean Spectral Bandwidth', value=1674.037)
std_spectral_bandwidth = st.number_input('std spectral bandwidth', value=407.410)	
mean_spectral_contrast	= st.number_input('Mean Spectral Contrast', value=21.480)
mean_spectral_flatness	= st.number_input('Mean Spectral Flatness', value=0.0251)
mean_spectral_rolloff	= st.number_input('Mean Spectral Rolloff', value=3927.989)
zero_crossing_rate	= st.number_input('Zero Crossing Rate', value=0.187)
rms_energy = st.number_input('RMS Energy', value=0.082)
mean_pitch = st.number_input('Mean Pitch', value=1406.35)
min_pitch	= st.number_input('Min Pitch', value=156.848)
max_pitch = st.number_input('Max Pitch', value=3994.698)	
std_pitch = st.number_input('Std Pitch', value=1139.946)	
spectral_skew = st.number_input('spectral Skew', value=1.417)
spectral_kurtosis = st.number_input('spectral Kurtoise', value=0.829)	
energy_entropy = st.number_input('Energy Entropy', value=30.805)	
log_energy = st.number_input('Log energy', value=2.244)
mfcc_1_mean = st.number_input('mfcc 1 mean', value=-239.697)	
mfcc_1_std	= st.number_input('mfcc 1 std', value=1450.0)
mfcc_2_mean = st.number_input('mfcc 2 mean', value=1450.0)	
mfcc_2_std = st.number_input('mfcc 2 std', value=1450.0)
mfcc_3_mean	= st.number_input('mfcc 3 mean', value=1450.0)
mfcc_3_std	= st.number_input('mfcc 3 std', value=1450.0)
mfcc_4_mean	= st.number_input('mfcc 4 mean', value=1450.0)
mfcc_4_std = st.number_input('mfcc 4 std', value=1450.0)	
mfcc_5_mean = st.number_input('mfcc 5 mean', value=1450.0)
mfcc_5_std	= st.number_input('mfcc 5 std', value=1450.0)
mfcc_6_mean	= st.number_input('mfcc 6 mean', value=1450.0)
mfcc_6_std	= st.number_input('mfcc 6 std', value=1450.0)
mfcc_7_mean	= st.number_input('mfcc 7 mean', value=1450.0)
mfcc_7_std	= st.number_input('mfcc 7 std', value=1450.0)
mfcc_8_mean	= st.number_input('mfcc 8 mean', value=1450.0)
mfcc_8_std	= st.number_input('mfcc 8 std', value=1450.0)
mfcc_9_mean	= st.number_input('mfcc 9 mean', value=1450.0)
mfcc_9_std	= st.number_input('mfcc 9 std', value=1450.0)
mfcc_10_mean = st.number_input('mfcc 10 mean', value=1450.0)
mfcc_10_std	= st.number_input('mfcc 10 std', value=1450.0)
mfcc_11_mean = st.number_input('mfcc 11 mean', value=1450.0)
mfcc_11_std	= st.number_input('mfcc 11 std', value=1450.0)
mfcc_12_mean = st.number_input('mfcc 12 mean', value=1450.0)
mfcc_12_std	= st.number_input('mfcc 12 std', value=1450.0)
mfcc_13_mean = st.number_input('mfcc 13 mean', value=1450.0)
mfcc_13_std = st.number_input('mfcc 13 std', value=1450.0)



# Collect inputs into DataFrame for prediction
input_data = pd.DataFrame({
    'mean_spectral_centroid' : [mean_spectral_centroid],	
    'std_spectral_centroid'	 : [std_spectral_centroid],
    'mean_spectral_bandwidth' : [mean_spectral_bandwidth],	
    'std_spectral_bandwidth' : [std_spectral_bandwidth],
    'mean_spectral_contrast' : [mean_spectral_contrast],	
    'mean_spectral_flatness' : [mean_spectral_flatness],
    'mean_spectral_rolloff' : [mean_spectral_rolloff],	
    'zero_crossing_rate' : [zero_crossing_rate],	
    'rms_energy' : [rms_energy],	
    'mean_pitch' : 	[mean_pitch],
    'min_pitch' : [min_pitch],	
    'max_pitch' : [max_pitch],	
    'std_pitch' : [std_pitch],	
    'spectral_skew' : [spectral_skew],	
    'spectral_kurtosis' : [spectral_kurtosis],	
    'energy_entropy' : [energy_entropy],	
    'log_energy' : [log_energy],	
    'mfcc_1_mean' : [mfcc_1_mean],	
    'mfcc_1_std' : [mfcc_1_std],
    'mfcc_2_mean' : [mfcc_2_mean],	
    'mfcc_2_std' : [mfcc_2_std],	
    'mfcc_3_mean' : [mfcc_3_mean],		
    'mfcc_3_std' : [mfcc_3_std],	
    'mfcc_4_mean' : [mfcc_4_mean],	
    'mfcc_4_std' : [mfcc_4_std],	
    'mfcc_5_mean' : [mfcc_5_mean],	
    'mfcc_5_std' : [mfcc_5_std],	
    'mfcc_6_mean' : [mfcc_6_mean],	
    'mfcc_6_std' : [mfcc_6_std],	
    'mfcc_7_mean' : [mfcc_7_mean],	
    'mfcc_7_std' : [mfcc_7_std],	
    'mfcc_8_mean' : [mfcc_8_mean],
    'mfcc_8_std' : [mfcc_8_std],	
    'mfcc_9_mean' : [mfcc_9_mean],	
    'mfcc_9_std' : [mfcc_9_std],	
    'mfcc_10_mean' : [mfcc_10_mean],	
    'mfcc_10_std' : [mfcc_10_std],	
    'mfcc_11_mean' : [mfcc_11_mean],	
    'mfcc_11_std' : [mfcc_11_std],	
    'mfcc_12_mean' : [mfcc_12_mean],	
    'mfcc_12_std' : [mfcc_12_std],	
    'mfcc_13_mean' : [mfcc_13_mean],	
    'mfcc_13_std' : [mfcc_13_std]


})

if st.button('Predict Gender'):
    prediction = rfcl.predict(input_data)
    label = 'Male' if prediction[0] == 1 else 'Female'
    st.success(f'The predicted gender is: {label}')

# Optional: Show some dataset insights (load your dataset)
if st.checkbox('Show Dataset Summary'):
    voice_data = pd.read_csv("vocal_gender_features_new.csv")
    st.write(voice_data.describe())
    st.write("Class distribution:")
    st.write(voice_data['label'].value_counts())
