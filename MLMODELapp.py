import streamlit as st
import joblib
import os
import numpy as np
import base64
import sklearn
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.impute import SimpleImputer

attrib_info = """
#### Attribute Information:
    - Age 1.20-65
    - SkinThickness (mm/dl) Numerical
    - Glucose level (mm/dl) Numerical
    - Insulin (mm/dl) Numerical
    - BMI (mm/dl) Numerical
    - Blood pressure (mm/Hg) Numerical
    - Activity tracker 1.Yes, 2.No.
    - Outcome 1.Positive, 2.Negative.
"""
label_dict = {"No": 0, "Yes": 1}
target_label_map = {"Negative": 0, "Positive": 1}

['age', 'SkinThickness', 'glucose_level', 'BMI', 'Insulin', 'blood_pressure', 'activity_tracker', 'Outcome']


def get_fvalue(val):
    feature_dict = {"No": 0, "Yes": 1}
    return feature_dict.get(val, val)


def get_value(val, my_dict):
    return next((key for key, value in my_dict.items() if value == val), val)


#@st.cache
#def load_model(model_file):
    #loaded_model = joblib.load(open(os.path.join(model_file), "rb"))
   # return loaded_model


def run_ml_app():
    st.subheader("Machine Learning Section")
    #loaded_model = load_model("diabetes_model.pkl")

    with st.expander("Attributes Info"):
        st.markdown(attrib_info, unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        age = st.number_input("Age", 10, 100)
        #glucose_level = st.number_input("Glucose Level")
       # glucose_level=st.input("https://thingspeak.com/channels/2382017/widgets/771608")
        glucose_level = st.number_input("Glucose Level", value=0.0)
        blood_pressure = st.number_input("Blood Pressure")
        SkinThickness = st.number_input("SkinThickness")

    with col2:
        BMI = st.number_input("BMI")
        Insulin = st.number_input("Insulin")
        activity_tracker = st.radio("Activity Tracker", ["No", "Yes"])

    st.subheader("View Your Selected Options here")
    with st.expander("Your Selected Options"):
        result = {
            'age': age,
            'glucose_level': glucose_level,
            'SkinThickness': SkinThickness,
            'blood_pressure': blood_pressure,
            "BMI": BMI,
            "Insulin": Insulin,
            'activity_tracker': activity_tracker
        }
        st.write(result)
        encoded_result = [get_fvalue(val) for val in result.values()]

    st.subheader("View Your Diagnosis Report Here")
    with st.expander("Predicted Results"):
        single_sample = np.array(encoded_result).reshape(1, -1)

        # Check for non-finite values in the input array
        if np.isnan(single_sample).any() or np.isinf(single_sample).any():
            st.warning("Please provide valid input values.")
        else:
             X = np.array([[np.nan, 2, 125, 26, 185, 75], [6, np.nan, 105, 32, 250, 80], [7, 6, 85, 28, np.nan, 70]])
             y = np.array([1, 0, 1])  # Replace with your target variable

            # Replace the NaN values in X with appropriate values using an imputer
             imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
             X_imputed = imputer.fit_transform(X)

            # Create a DecisionTreeClassifier and fit the imputed data
             classifier = DecisionTreeClassifier()
             classifier.fit(X_imputed, y)

             # Transform the single sample using the imputer
             single_sample_imputed = imputer.transform(single_sample[:, :-1])

             # Predict the outcome for the single sample
             prediction = classifier.predict(single_sample_imputed)
             pred_prob = classifier.predict_proba(single_sample_imputed)
        if prediction == 1:
            st.warning("Positive Risk-{}".format(prediction[0]))
            pred_probability_score = {"Negative DM": pred_prob[0][0] * 100, "Positive DM": pred_prob[0][1] * 100}
            st.subheader("Prediction Probability Score")
            st.json(pred_probability_score)
            st.subheader(
                f"You are Likely to have diabetics, we estimated there is {round(pred_prob[0][1] * 99, 4)}% of chance of you having diabteics")

        else:
            st.success("Negative Risk-{}".format(prediction[0]))
            pred_probability_score = {"Negative DM": pred_prob[0][0] * 100, "Positive DM": pred_prob[0][1] * 100}
            st.subheader("Prediction Probability Score")
            st.json(pred_probability_score)
            st.subheader(
                f"Woohoo!, You don't have a risk of diabetics, but we estimated there is {round(pred_prob[0][1] * 100, 4)}% of chance of you having diabteics. Be Careful ! Take Care! ")
