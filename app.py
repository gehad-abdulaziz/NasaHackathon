import streamlit as st
import joblib
import pandas as pd
import numpy as np

# -----------------
# 1. Model Loading and Caching
# -----------------

@st.cache_resource # Cache the model to ensure it is loaded only once
def load_model():
    """Load the saved model from the PKL file"""
    try:
        # ✅ Updated file name to 'best_tuned_exoplanet_classifier_model.pkl'
        model = joblib.load('best_tuned_exoplanet_classifier_model.pkl')
        return model
    except FileNotFoundError:
        # Clear error message
        st.error("❌ ERROR: Model file 'best_tuned_exoplanet_classifier_model.pkl' not found. Ensure it is in the same directory.")
        return None

# Load the model
model = load_model()

# -----------------
# 2. Feature Engineering Function (Applying the same transformations)
# -----------------

# ✅ Final corrected feature column order to match the trained model: st_mass precedes st_logg
FEATURE_COLUMNS = [
    'transit_duration', 'eq_temp', 'st_teff', 'st_rad', 
    'st_mass',             # ⬅️ This must be before st_logg
    'st_logg',             # ⬅️ This must be after st_mass
    'st_met', 'st_dist', 'st_mag', 'transit_depth_log', 
    'planet_radius_log', 'insolation_log', 
    'radius_ratio',        # ⬅️ This must be before log_orbital_period
    'log_orbital_period',  # ⬅️ This must be after radius_ratio
    'normalized_transit_depth'
]

def apply_feature_engineering(input_data):
    """Apply the exact same transformations and feature engineering performed during training"""
    
    # Convert input data to a DataFrame
    df = pd.DataFrame([input_data])
    
    # 1. Apply Logarithmic Transformation
    df['transit_depth_log'] = np.log1p(df['transit_depth'])
    df['planet_radius_log'] = np.log1p(df['planet_radius'])
    df['insolation_log'] = np.log1p(df['insolation'])
    df['log_orbital_period'] = np.log10(df['orbital_period'] + 1e-5)
    
    # 2. New Feature Engineering
    df['radius_ratio'] = df['planet_radius'] / df['st_rad']
    df['normalized_transit_depth'] = df['transit_depth'] / df['st_mag']
    
    # 3. Return columns in the correct order (using FEATURE_COLUMNS)
    return df[FEATURE_COLUMNS]

# -----------------
# 3. Streamlit User Interface
# -----------------

st.title("🛰️ Exoplanet Classifier")
st.markdown("Enter the astronomical parameters of the candidate to predict its classification: **Confirmed, Candidate, or False Positive**.")

if model is not None:
    # Divide the screen into 3 columns to organize inputs
    col1, col2, col3 = st.columns(3)
    
    # Input fields for Planet and Star data
    with col1:
        st.header("Planet Parameters 🪐")
        p_rad = st.number_input("Planet Radius [R_Earth]", min_value=0.01, value=1.0, step=0.01)
        t_dur = st.number_input("Transit Duration [Hours]", min_value=0.01, value=3.0, step=0.1)
        t_depth = st.number_input("Transit Depth [ppm]", min_value=0.01, value=500.0, step=1.0)
        p_orb = st.number_input("Orbital Period [Days]", min_value=0.01, value=10.0, step=0.1)

    with col2:
        st.header("Host Star Parameters ⭐")
        s_teff = st.number_input("Stellar Temperature [K]", min_value=1000, value=5700, step=10)
        s_rad = st.number_input("Stellar Radius [Rsun]", min_value=0.1, value=1.0, step=0.01)
        s_logg = st.number_input("Stellar Logg (Surface Gravity)", min_value=0.0, value=4.4, step=0.01)
        s_mass = st.number_input("Stellar Mass [Msun]", min_value=0.1, value=1.0, step=0.01)
        
    with col3:
        st.header("Additional Parameters 🔭")
        s_met = st.number_input("Stellar Metallicity", value=0.0, step=0.01)
        s_dist = st.number_input("Stellar Distance [pc]", min_value=0.0, value=100.0, step=1.0)
        s_mag = st.number_input("Stellar Magnitude (Mag)", min_value=5.0, value=12.0, step=0.1)
        eq_temp = st.number_input("Equilibrium Temperature [K]", min_value=0, value=500, step=1)
        insolation = st.number_input("Insolation", min_value=0.01, value=1.0, step=0.01)


    # Collect inputs into a dictionary
    input_data = {
        'transit_duration': t_dur, 
        'eq_temp': eq_temp, 
        'st_teff': s_teff, 
        'st_rad': s_rad, 
        'st_logg': s_logg, 
        'st_mass': s_mass, 
        'st_met': s_met, 
        'st_dist': s_dist, 
        'st_mag': s_mag,
        'transit_depth': t_depth,
        'planet_radius': p_rad,
        'insolation': insolation,
        'orbital_period': p_orb,
    }

    # -----------------
    # 4. Prediction
    # -----------------
    
    if st.button("Predict Classification", type="primary"):
        # Apply feature engineering
        processed_input = apply_feature_engineering(input_data)
        
        # Prediction
        prediction_num = model.predict(processed_input)[0]
        
        # Convert numerical prediction to a text label
        label_mapping = {
            0: "❌ FALSE POSITIVE (Not a planet)",
            1: "⭐ CANDIDATE (Needs confirmation)",
            2: "✅ CONFIRMED (Real planet)"
        }
        prediction_label = label_mapping.get(prediction_num, "Unknown")
        
        st.markdown("---")
        st.subheader("Prediction Results:")
        
        if prediction_num == 2:
            st.success(f"Model Predicts: **{prediction_label}**")
        elif prediction_num == 1:
            st.warning(f"Model Predicts: **{prediction_label}**")
        else:
            st.error(f"Model Predicts: **{prediction_label}**")


# -----------------
# How to Run
# -----------------
st.markdown("---")
st.code("To run the application, ensure the model file is in the same directory, then type in the terminal:\n\nstreamlit run app.py")
