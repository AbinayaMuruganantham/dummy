import streamlit as st
import streamlit.components.v1 as stc
from EDAapp import run_eda_app
from MLMODELapp import run_ml_app
from PIL import Image
import webbrowser

html_temp = """
    <div style="background-color:#35858B;padding:10px;border-radius:10px">
    <h1 style="color:#AEFEFF;text-align:center;">DIABETANET: AN INTELLIGENT ECOSYSTEM FOR PREDICTIVE DIABETES MONITORING VIA CLOUD</h1>
    <h2 style="color:#064663;text-align:center;"> </h2>
    </div>
    """

def main():
    st.set_page_config(page_title="DESIGN PROJECT DEMO", page_icon='🍫', layout="wide")
    stc.html(html_temp)
    st.markdown("<p><TT>Designed and Developed by <a style='text-decoration:none;color:red' target='_blank'> Abinaya M , Aaathyuktha S , Sruthi S </a></TT></p>", unsafe_allow_html=True)

    menu = ["Diabetic Diagnosis", "EDA", "About"]
    st.sidebar.write("1. Select EDA option to see detailed analysis of the datset")
    st.sidebar.write("2. Select Diabetic Diagnosis to use Diabetes Risk Predictor")
    choice = st.sidebar.selectbox("Choose One of the Option", menu)

    if choice == "Home":
        st.subheader("Home")
        st.write("""
            This dataset contains the sign and symptoms data of newly diabetic or would be diabetic patient.
            #### Datasource
                - https://archive.ics.uci.edu/ml/datasets/Early+stage+diabetes+risk+prediction+dataset.
            #### App Content
                - EDA Section: Exploratory Data Analysis of Data
                - ML Section: ML Predictor App
            """)
    elif choice == "EDA":
        run_eda_app()
    elif choice == "Diabetic Diagnosis":
        run_ml_app()
    else:
        image = Image.open('preview/dp.jpg')
        col1, col2 = st.columns([1, 2])
        with col1:
            st.image(image)

        with col2:
            st.markdown("<h1> Hey There <span style='display: block;'> I'm Abinaya Muruganantham</span> </h1>", unsafe_allow_html=True)
            st.markdown("<h5>I have skills in AWS - PYTHON - JAVA FULL STACK DEVELOPMENT - MACHINE LEARNING <h5>", unsafe_allow_html=True)
            st.write("A self-motivated, highly passionate, and hardworking person looking for an opportunity to work in a challenging organization to utilize my skills and knowledge for the growth of the organization")
            st.text("Connect With ME : ")
            github = st.button("Visit My Github")
            linkedin = st.button("Visit My Linkedin")
            if github:
                webbrowser.open('https://github.com/AbinayaMuruganantham/')
            if linkedin:
                webbrowser.open('https://www.linkedin.com/in/abinaya-muruganantham-0a2880222/?originalSubdomain=in')

if __name__ == '__main__':
    main()
