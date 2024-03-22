import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load the dataset
diabetes_data = pd.read_csv(r'diabetes.csv')

def run_eda_app():
    st.title("Diabetes Prediction - Exploratory Data Analysis")

    # Show the dataset
    st.subheader("Diabetes Dataset")
    st.dataframe(diabetes_data)

    # Show basic statistics
    st.subheader("Basic Statistics")
    st.write(diabetes_data.describe())

    # Show correlation heatmap
    st.subheader("Correlation Heatmap")
    plt.figure(figsize=(10, 8))
    sns.heatmap(diabetes_data.corr(), annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
    st.pyplot()

    # Show distribution of target variable

