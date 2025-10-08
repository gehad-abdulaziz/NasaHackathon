# 🌟 Exoplanet Classification Project

## 📋 Project Overview

This project implements a machine learning model to classify exoplanets into three categories: **FALSE POSITIVE**, **CANDIDATE**, and **CONFIRMED**. The model uses data from three major exoplanet discovery missions: Kepler (cumulative), K2, and TESS (TOI), combining over 21,000 observations to train an optimized XGBoost classifier.

## 🎯 Key Results

- **Final Model Accuracy: 77.93%**
- **Best Cross-Validation Score: 78.78%**
- **Model Type:** XGBoost Classifier with Randomized Hyperparameter Tuning

### Performance Breakdown
| Class | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|---------|
| FALSE POSITIVE | 0.79 | 0.76 | 0.77 | 1,111 |
| CANDIDATE | 0.79 | 0.72 | 0.76 | 1,642 |
| CONFIRMED | 0.76 | 0.86 | 0.81 | 1,502 |

## 📊 Dataset Information

### Data Sources
1. **Cumulative Dataset** (Kepler): 9,564 observations × 49 features
2. **K2 PandC Dataset**: 4,004 observations × 94 features
3. **TOI Dataset** (TESS): 7,703 observations × 65 features

**Total Combined Dataset:** 21,271 observations after merging

### Features Used
The model utilizes 15 standardized features across all datasets:

**Planetary Features:**
- `orbital_period` - Orbital period (days)
- `transit_duration` - Transit duration (hours)
- `transit_depth` - Transit depth (ppm)
- `planet_radius` - Planet radius (Earth radii)
- `eq_temp` - Equilibrium temperature (K)
- `insolation` - Stellar insolation flux

**Stellar Features:**
- `st_teff` - Stellar effective temperature (K)
- `st_rad` - Stellar radius (solar radii)
- `st_mass` - Stellar mass (solar masses)
- `st_logg` - Stellar surface gravity (log g)
- `st_met` - Stellar metallicity [Fe/H]
- `st_dist` - Distance to star (parsecs)
- `st_mag` - Stellar magnitude

**Engineered Features:**
- `transit_depth_log` - Log-transformed transit depth
- `planet_radius_log` - Log-transformed planet radius
- `insolation_log` - Log-transformed insolation
- `radius_ratio` - Planet-to-star radius ratio
- `log_orbital_period` - Log-transformed orbital period
- `normalized_transit_depth` - Transit depth normalized by stellar magnitude

## 🔧 Data Processing Pipeline

### 1. Data Loading & Merging
- Loaded three separate CSV files from NASA Exoplanet Archive
- Standardized column names across datasets
- Merged into unified dataset with common feature schema

### 2. Missing Data Handling
**Initial Missing Data Rates:**
- `planet_mass`: 97.99% → **Dropped** (insufficient data)
- `st_met`: 92.05% → Imputed
- `st_mass`: 90.18% → Imputed
- Other features: <52% missing → Imputed

**Imputation Strategy:**
- **KNN Imputer** (k=5) for most numeric features
- **Random Forest Regressor** for stellar mass and metallicity using correlated features

### 3. Feature Engineering
- Log transformations for skewed distributions
- Created derived features (ratios, normalized values)
- Removed highly collinear features

### 4. Class Distribution
- Class 1 (CANDIDATE): 37.09%
- Class 2 (CONFIRMED): 36.40%
- Class 0 (FALSE POSITIVE): 26.51%

Dataset is relatively balanced, no resampling required.

## 🤖 Model Development

### Base Model
- **Algorithm:** XGBoost Classifier
- **Objective:** Multi-class classification (3 classes)
- **Early Stopping:** 50 rounds
- **Training Rounds:** 532 iterations (stopped early)

### Hyperparameter Tuning
**Method:** Randomized Search CV (50 iterations, 5-fold CV)

**Optimal Parameters:**
```python
{
    'n_estimators': 500,
    'max_depth': 7,
    'learning_rate': 0.05,
    'gamma': 0.5,
    'subsample': 0.7,
    'colsample_bytree': 0.8
}
```

### Train/Validation/Test Split
- **Training Set:** 60% (12,762 samples)
- **Validation Set:** 20% (4,254 samples)
- **Test Set:** 20% (4,255 samples)

## 📈 Model Insights

### Feature Importance
The correlation analysis revealed:
- Strong correlations between planetary radius and transit depth
- Stellar properties (mass, radius, temperature) highly intercorrelated
- Orbital period shows moderate correlation with insolation flux

### Outlier Analysis
Outliers detected using IQR method:
- `insolation`: 3,135 outliers
- `orbital_period`: 2,982 outliers
- `transit_depth`: 2,391 outliers

Outliers retained as they represent valid astrophysical extremes.

## 💾 Model Deployment

### Saved Model
- **Filename:** `best_tuned_exoplanet_classifier_model.pkl`
- **Size:** 6.60 MB
- **Format:** Joblib pickle file
- **Dependencies:** XGBoost 3.0.5, scikit-learn, pandas, numpy

### Streamlit Application
A separate deployment file (`app.py`) provides:
- Interactive web interface for predictions
- Real-time classification of exoplanet candidates
- Visualization of prediction probabilities
- Feature importance display

## 🚀 Usage

### Requirements
```bash
pip install xgboost==3.0.5 scikit-learn pandas numpy matplotlib seaborn joblib streamlit
```

### Loading the Model
```python
import joblib

# Load the trained model
model = joblib.load('best_tuned_exoplanet_classifier_model.pkl')

# Make predictions
predictions = model.predict(X_new)
probabilities = model.predict_proba(X_new)
```

### Running Streamlit App
```bash
streamlit run app.py
```

## 📚 Project Structure
```
exoplanet-classification/
├── datasets/
│   ├── cumulative_2025.10.03_02.50.27.csv
│   ├── k2pandc_2025.10.03_02.47.49.csv
│   └── TOI_2025.10.03_02.47.18.csv
├── best_tuned_exoplanet_classifier_model.pkl
├── training_notebook.ipynb
├── app.py
└── README.md
```

## 🔬 Methodology Highlights

1. **Robust Data Integration:** Successfully merged heterogeneous datasets from different missions
2. **Advanced Imputation:** Combined KNN and ML-based imputation for missing values
3. **Feature Engineering:** Created domain-specific derived features based on astrophysical principles
4. **Comprehensive Tuning:** Exhaustive hyperparameter search with cross-validation
5. **Production-Ready:** Model saved and ready for deployment with Streamlit interface

## 📊 Visualizations Included

- Correlation heatmaps
- Feature distribution histograms
- Class distribution pie charts
- Pairplot analysis of key features
- Boxplots for outlier detection

## 🎓 Scientific Context

This model assists astronomers in:
- Prioritizing follow-up observations
- Filtering false positive detections
- Accelerating exoplanet confirmation process
- Understanding feature importance in classification

## 📖 References

- NASA Exoplanet Archive
- Kepler Mission Data
- K2 Mission Data
- TESS Objects of Interest (TOI)

## 👨‍💻 Technical Notes

- All numeric features scaled and normalized
- Class encoding: 0 (FALSE POSITIVE), 1 (CANDIDATE), 2 (CONFIRMED)
- Model trained on Google Colab with GPU acceleration
- Reproducibility ensured with `random_state=42`

---

**Model Version:** 1.0  
**Last Updated:** October 2025  
**Training Date:** October 3, 2025
## 🔬 Model Photos
![Model Diagram 3](https://drive.google.com/uc?export=view&id=1LqLTBU_nygtPQgN5yRbLbnD3Ot-1hY_7)
![Model Diagram 2](https://drive.google.com/uc?export=view&id=1Wev3rrFpYjnIlr91ih69_uHXr04MiNUC)
![Model Diagram 1](https://drive.google.com/uc?export=view&id=1txxk53lWKdf_tNodH7NKd6q4g_Z515FC)
ط
## you can try it from here :https://nasaspace-hpzv8cgyepxswaclf2pgqt.streamlit.app/
## or from here : https://rfaiutt873loqsxwvuacam.streamlit.app/ (May NOT work)
## my Certificate:https://drive.google.com/file/d/1Q4JLNdm74SZaKfp4r5N3ubo1va3oCF84/view?usp=sharing

# CosmoCrafters TEAM <3
