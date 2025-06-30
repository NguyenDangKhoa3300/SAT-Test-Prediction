import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import numpy as np

# ==== Cấu hình giao diện ====
st.set_page_config(page_title='SAT Math Prediction', layout='centered')

st.title('SAT Math Difficulty Prediction')
st.markdown('Input question and multiple choices to predict the difficulty.')

# ==== Load mô hình và preprocessor ====
pipeline = joblib.load('D:\\Self Study\Python\\Data Science\\Machine Learning\\Supervised machine learning\\OpenSAT\\sat_math_prediction_model.pkl')
model = pipeline.named_steps['classifier']

# ==== Form người dùng ====
with st.form('form_input'):
    question = st.text_area('Question', height=100)
    choice_a = st.text_input('Choice A')
    choice_b = st.text_input('Choice B')
    choice_c = st.text_input('Choice C')
    choice_d = st.text_input('Choice D')
    submitted = st.form_submit_button('Predict')

# ==== Xử lý khi Submit ====
if submitted:
    input_df = pd.DataFrame([{
        'question': question,
        'choice a': choice_a,
        'choice b': choice_b,
        'choice c': choice_c,
        'choice d': choice_d
    }])

    # Dự đoán
    pred = pipeline.predict(input_df)[0]
    prob = pipeline.predict_proba(input_df)[0]

    if pred == 'Hard':
        label = 'Hard'
        color = 'red'
    elif pred == 'Medium':
        label = 'Medium'
        color = 'orange'
    else:
        label = 'Easy'
        color = 'green'

    st.subheader('Prediction Result')
    st.markdown(f"<h3 style='color:{color}'>{label}</h3>", unsafe_allow_html=True)

    # Lấy xác suất đúng với nhãn dự đoán
    class_index = list(pipeline.classes_).index(pred)
    prob = pipeline.predict_proba(input_df)[0][class_index]
    st.metric("Xác suất độ khó", f"{prob:.2%}")

    if 'difficulty_probs' not in st.session_state:
        st.session_state.difficulty_probs = []
    st.session_state.difficulty_probs.append(prob)
