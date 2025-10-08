import streamlit as st
import joblib
import pandas as pd
import numpy as np

# -----------------
# 1. Page Configuration & Custom Theme
# -----------------
st.set_page_config(
    page_title="Exoplanet Classifier",
    page_icon="🛰️",
    layout="wide"
)

# Custom CSS for dark neon theme
st.markdown("""
    <style>
        /* General Page Styling */
        body, .stApp {
            background-color: #0a1128;
            color: #e6e6e6;
            font-family: 'Poppins', sans-serif;
        }

        /* Headings */
        h1, h2, h3, h4 {
            color: #ffffff !important;
            text-align: center;
            font-weight: 600;
        }

        /* Section titles (subheaders) */
        .stSubheader {
            color: #ff4b5c !important;
            font-weight: bold !important;
        }

        /* Card Containers */
        div[data-testid="stVerticalBlock"] > div {
            background: linear-gradient(145deg, #101d42, #0b132b);
            border-radius: 20px;
            padding: 25px;
            margin-bottom: 25px;
            box-shadow:
                0 0 10px rgba(255, 0, 85, 0.3),
                0 0 20px rgba(0, 150, 255, 0.2),
                inset 0 0 10px rgba(255, 255, 255, 0.05);
            transition: all 0.3s ease-in-out;
        }

        /* Hover effect for glowing cards */
        div[data-testid="stVerticalBlock"] > div:hover {
            box-shadow:
                0 0 25px rgba(255, 0, 85, 0.6),
                0 0 35px rgba(0, 150, 255, 0.5),
                inset 0 0 10px rgba(255, 255, 255, 0.1);
            transform: scale(1.02);
        }

        /* Input labels */
        label {
            color: #e0e0e0 !important;
            font-weight: 500 !important;
        }

        /* Input fields */
        .stNumberInput input {
            background-color: #1c2541 !important;
            color: #ffffff !important;
            border-radius: 10px;
        }

        /* Buttons */
        button[kind="primary"] {
            background: linear-gradient(90deg, #ff0033, #ff4b5c);
            color: white !important;
            border: none !important;
            border-radius: 12px !important;
            font-size: 18px !important;
            font-weight: 600;
            padding: 10px 25px;
            transition: 0.3s ease;
            box-shadow: 0 0 15px rgba(255, 0, 55, 0.4);
        }
        button[kind="primary"]:hover {
            background: linear-gradient(90deg, #ff4b5c, #ff0033);
            box-shadow: 0 0 25px rgba(255, 0, 85, 0.8);
            transform: scale(1.05);
        }

        /* Code box */
        .stCodeBlock {
            background-color: #1b1b2f !important;
            color: #ffffff !important;
        }

        /* Prediction boxes */
        .stSuccess {
            background-color: #0b3d20 !important;
            color: #00ff99 !important;
            border-left: 5px solid #00ff99 !important;
            border-radius: 10px;
            box-shadow: 0 0 20px rgba(0,255,153,0.4);
        }
        .stWarning {
            background-color: #3a506b !important;
            color: #ffcc00 !important;
            border-left: 5px solid #ffcc00 !important;
            border-radius: 10px;
            box-shadow: 0 0 20px rgba(255,204,0,0.4);
        }
        .stError {
            background-color: #3a0ca3 !important;
            color: #ff0033 !important;
            border-left: 5px solid #ff0033 !important;
            border-radius: 10px;
            box-shadow: 0 0 25px rgba(255,0,85,0.5);
        }
    </style>
""", unsafe_allow_html=True)

# -----------------
# 2. Model Loading
# -----------------
@st.cache_resource
def load_model():
    try:
        model = joblib.load('best_tuned_exoplanet_classifier_model.pkl')
        return model
    except FileNotFoundError:
        st.error("❌ Model file not found. Make sure it's in the same directory.")
        return None

model = load_model()

# -----------------
# 3. Feature Engineering
# -----------------
FEATURE_COLUMNS = [
    'transit_duration', 'eq_temp', 'st_teff', 'st_rad',
    'st_mass', 'st_logg', 'st_met', 'st_dist', 'st_mag',
    'transit_depth_log', 'planet_radius_log', 'insolation_log',
    'radius_ratio', 'log_orbital_period', 'normalized_transit_depth'
]

def apply_feature_engineering(input_data):
    df = pd.DataFrame([input_data])
    df['transit_depth_log'] = np.log1p(df['transit_depth'])
    df['planet_radius_log'] = np.log1p(df['planet_radius'])
    df['insolation_log'] = np.log1p(df['insolation'])
    df['log_orbital_period'] = np.log10(df['orbital_period'] + 1e-5)
    df['radius_ratio'] = df['planet_radius'] / df['st_rad']
    df['normalized_transit_depth'] = df['transit_depth'] / df['st_mag']
    return df[FEATURE_COLUMNS]

# -----------------
# 4. UI Layout
# -----------------
st.title("🛰️ Exoplanet Classifier")
st.markdown("<h4 style='color:#b3b3ff'>Identify if a celestial object is a Confirmed Planet, Candidate, or False Positive.</h4>", unsafe_allow_html=True)
st.divider()

if model is not None:
    col1, col2, col3 = st.columns(3)

    with col1:
        st.subheader("🪐 Planet Parameters")
        p_rad = st.number_input("Planet Radius [R_Earth]", min_value=0.01, value=1.0, step=0.01)
        t_dur = st.number_input("Transit Duration [Hours]", min_value=0.01, value=3.0, step=0.1)
        t_depth = st.number_input("Transit Depth [ppm]", min_value=0.01, value=500.0, step=1.0)
        p_orb = st.number_input("Orbital Period [Days]", min_value=0.01, value=10.0, step=0.1)

    with col2:
        st.subheader("⭐ Host Star Parameters")
        s_teff = st.number_input("Stellar Temperature [K]", min_value=1000, value=5700, step=10)
        s_rad = st.number_input("Stellar Radius [Rsun]", min_value=0.1, value=1.0, step=0.01)
        s_logg = st.number_input("Stellar Logg (Surface Gravity)", min_value=0.0, value=4.4, step=0.01)
        s_mass = st.number_input("Stellar Mass [Msun]", min_value=0.1, value=1.0, step=0.01)

    with col3:
        st.subheader("🔭 Additional Parameters")
        s_met = st.number_input("Stellar Metallicity", value=0.0, step=0.01)
        s_dist = st.number_input("Stellar Distance [pc]", min_value=0.0, value=100.0, step=1.0)
        s_mag = st.number_input("Stellar Magnitude (Mag)", min_value=5.0, value=12.0, step=0.1)
        eq_temp = st.number_input("Equilibrium Temperature [K]", min_value=0, value=500, step=1)
        insolation = st.number_input("Insolation", min_value=0.01, value=1.0, step=0.01)

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

    if st.button("🚀 Predict Classification", type="primary"):
        processed_input = apply_feature_engineering(input_data)
        prediction_num = model.predict(processed_input)[0]

        label_mapping = {
            0: "❌ FALSE POSITIVE (Not a planet)",
            1: "⭐ CANDIDATE (Needs confirmation)",
            2: "✅ CONFIRMED (Real planet)"
        }
        prediction_label = label_mapping.get(prediction_num, "Unknown")

        st.markdown("---")
        st.subheader("🧠 Prediction Results")

        if prediction_num == 2:
            st.success(f"Model Predicts: **{prediction_label}**")
        elif prediction_num == 1:
            st.warning(f"Model Predicts: **{prediction_label}**")
        else:
            st.error(f"Model Predicts: **{prediction_label}**")

st.markdown("---")
st.code("Run the app:\n\npython -m streamlit run app.py", language="bash")
