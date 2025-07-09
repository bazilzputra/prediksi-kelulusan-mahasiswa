import streamlit as st
import pandas as pd
import pickle

# ========================================
# 1. LOAD MODEL & ENCODER
# ========================================
model = pickle.load(open('rf_kelulusan_model.sav', 'rb'))
label_encoders = pickle.load(open('label_encoders.pkl', 'rb'))
le_status = pickle.load(open('le_status.pkl', 'rb'))

# ========================================
# 2. CONFIG
# ========================================
st.set_page_config(
    page_title="Prediksi Kelulusan Mahasiswa",
    page_icon="🎓",
    layout="centered"
)

st.title("🎓 Prediksi Tingkat Kelulusan Mahasiswa")

# ========================================
# 3. INPUT FORM
# ========================================
st.write("Isi data mahasiswa untuk memprediksi status kelulusan:")

col1, col2 = st.columns(2)

with col1:
    gender = st.selectbox("Gender", ["male", "female"])
    race = st.selectbox("Race/Ethnicity", ['group A', 'group B', 'group C', 'group D', 'group E'])
    parental_education = st.selectbox("Parental Level of Education",
                                      ["some high school", "high school", "some college",
                                       "associate's degree", "bachelor's degree", "master's degree"])

with col2:
    lunch = st.selectbox("Lunch", ["standard", "free/reduced"])
    prep_course = st.selectbox("Test Preparation Course", ["none", "completed"])
    math_score = st.number_input("Math Score", min_value=0, max_value=100, value=50)
    reading_score = st.number_input("Reading Score", min_value=0, max_value=100, value=50)
    writing_score = st.number_input("Writing Score", min_value=0, max_value=100, value=50)

if st.button("Prediksi"):
    input_dict = {
        'gender': gender,
        'race/ethnicity': race,
        'parental level of education': parental_education,
        'lunch': lunch,
        'test preparation course': prep_course,
        'math score': math_score,
        'reading score': reading_score,
        'writing score': writing_score
    }

    input_df = pd.DataFrame([input_dict])

    # Encoding
    for col in label_encoders:
        input_df[col] = label_encoders[col].transform(input_df[col])

    # Fitur akhir
    X_new = input_df

    # Prediksi
    pred = model.predict(X_new)
    pred_label = le_status.inverse_transform(pred)

    st.subheader("Hasil Prediksi:")
    st.success(f"Status Kelulusan: **{pred_label[0]}**")
