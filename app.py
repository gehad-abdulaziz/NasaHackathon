import streamlit as st
import joblib
import pandas as pd
import numpy as np

# -----------------
# 1. التحميل والتخزين المؤقت للنموذج (Cache Model Loading)
# -----------------

@st.cache_resource # لتخزين النموذج لضمان تحميله مرة واحدة فقط
def load_model():
    """تحميل النموذج المحفوظ من ملف PKL"""
    try:
        # ✅ تم تحديث اسم الملف إلى 'best_tuned_exoplanet_classifier_model.pkl'
        model = joblib.load('best_tuned_exoplanet_classifier_model.pkl')
        return model
    except FileNotFoundError:
        # رسالة خطأ واضحة
        st.error("❌ خطأ: لم يتم العثور على ملف النموذج 'best_tuned_exoplanet_classifier_model.pkl'. تأكد من وجوده في نفس المجلد.")
        return None

# تحميل النموذج
model = load_model()

# -----------------
# 2. دالة هندسة الميزات (تطبيق نفس التحويلات)
# -----------------

# ✅ تم تصحيح ترتيب الأعمدة النهائي ليطابق النموذج المُدرب: st_mass يسبق st_logg
FEATURE_COLUMNS = [
    'transit_duration', 'eq_temp', 'st_teff', 'st_rad', 
    'st_mass',             # ⬅️ يجب أن يكون هذا قبل st_logg
    'st_logg',             # ⬅️ يجب أن يكون هذا بعد st_mass
    'st_met', 'st_dist', 'st_mag', 'transit_depth_log', 
    'planet_radius_log', 'insolation_log', 
    'radius_ratio',        # ⬅️ يجب أن يكون هذا قبل log_orbital_period
    'log_orbital_period',  # ⬅️ يجب أن يكون هذا بعد radius_ratio
    'normalized_transit_depth'
]

def apply_feature_engineering(input_data):
    """تطبيق نفس تحويلات وهندسة الميزات التي تمت في مرحلة التدريب"""
    
    # تحويل البيانات إلى إطار بيانات (Dataframe)
    df = pd.DataFrame([input_data])
    
    # 1. تطبيق التحويل اللوغاريتمي (Log Transform)
    df['transit_depth_log'] = np.log1p(df['transit_depth'])
    df['planet_radius_log'] = np.log1p(df['planet_radius'])
    df['insolation_log'] = np.log1p(df['insolation'])
    df['log_orbital_period'] = np.log10(df['orbital_period'] + 1e-5)
    
    # 2. هندسة الميزات الجديدة
    df['radius_ratio'] = df['planet_radius'] / df['st_rad']
    df['normalized_transit_depth'] = df['transit_depth'] / df['st_mag']
    
    # 3. إرجاع الأعمدة بالترتيب الصحيح (باستخدام FEATURE_COLUMNS)
    return df[FEATURE_COLUMNS]

# -----------------
# 3. واجهة المستخدم في Streamlit
# -----------------

st.title("🛰️ مصنّف الكواكب الخارجية (Exoplanet Classifier)")
st.markdown("أدخل المعلمات الفلكية للمرشح للتنبؤ بتصنيفه: **Confirmed, Candidate, or False Positive**.")

if model is not None:
    # تقسيم الشاشة إلى 3 أعمدة لتنظيم المدخلات
    col1, col2, col3 = st.columns(3)
    
    # حقول إدخال بيانات الكوكب والنجم
    with col1:
        st.header("معلمات الكوكب 🪐")
        p_rad = st.number_input("نصف قطر الكوكب (Planet Radius) [R_Earth]", min_value=0.01, value=1.0, step=0.01)
        t_dur = st.number_input("مدة العبور (Transit Duration) [ساعات]", min_value=0.01, value=3.0, step=0.1)
        t_depth = st.number_input("عمق العبور (Transit Depth) [ppm]", min_value=0.01, value=500.0, step=1.0)
        p_orb = st.number_input("الفترة المدارية (Orbital Period) [أيام]", min_value=0.01, value=10.0, step=0.1)

    with col2:
        st.header("معلمات النجم المضيف ⭐")
        s_teff = st.number_input("درجة حرارة النجم (Stellar Temp) [K]", min_value=1000, value=5700, step=10)
        s_rad = st.number_input("نصف قطر النجم (Stellar Radius) [Rsun]", min_value=0.1, value=1.0, step=0.01)
        s_logg = st.number_input("جاذبية السطح (Stellar Logg)", min_value=0.0, value=4.4, step=0.01)
        s_mass = st.number_input("كتلة النجم (Stellar Mass) [Msun]", min_value=0.1, value=1.0, step=0.01)
        
    with col3:
        st.header("معلمات إضافية 🔭")
        s_met = st.number_input("معدنية النجم (Stellar Metallicity)", value=0.0, step=0.01)
        s_dist = st.number_input("مسافة النجم (Stellar Distance) [pc]", min_value=0.0, value=100.0, step=1.0)
        s_mag = st.number_input("سطوع النجم (Stellar Mag)", min_value=5.0, value=12.0, step=0.1)
        eq_temp = st.number_input("درجة حرارة التوازن (Eq Temp) [K]", min_value=0, value=500, step=1)
        insolation = st.number_input("الإشعاع الشمسي (Insolation)", min_value=0.01, value=1.0, step=0.01)


    # تجميع المدخلات في قاموس
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
    # 4. التوقع
    # -----------------
    
    if st.button("توقع التصنيف", type="primary"):
        # تطبيق هندسة الميزات
        processed_input = apply_feature_engineering(input_data)
        
        # التوقع
        prediction_num = model.predict(processed_input)[0]
        
        # تحويل التوقع الرقمي إلى تسمية نصية
        label_mapping = {
            0: "❌ FALSE POSITIVE (غير كوكب)",
            1: "⭐ CANDIDATE (مرشح)",
            2: "✅ CONFIRMED (مؤكد - كوكب حقيقي)"
        }
        prediction_label = label_mapping.get(prediction_num, "غير معروف")
        
        st.markdown("---")
        st.subheader("نتائج التنبؤ:")
        
        if prediction_num == 2:
            st.success(f"النموذج يتوقع: **{prediction_label}**")
        elif prediction_num == 1:
            st.warning(f"النموذج يتوقع: **{prediction_label}**")
        else:
            st.error(f"النموذج يتوقع: **{prediction_label}**")


# -----------------
# كيفية التشغيل
# -----------------
st.markdown("---")
st.code("لتشغيل التطبيق، تأكد من وجود ملف النموذج في نفس المجلد ثم اكتب في الطرفية:\n\nstreamlit run app.py")
