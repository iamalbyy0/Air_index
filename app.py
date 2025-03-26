import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image

st.title("Monster")
st.title("Winter Is Coming...")

# File uploader
uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

if uploaded_file is not None:
    st.write("File Uploaded..Success.")
    
    # Load and display data INSIDE the if block
    df = pd.read_csv(uploaded_file)
    st.subheader("Data preview")
    st.write(df.head())
    
    st.subheader('Data Summary')
    st.write(df.describe())

# Move image outside so it shows even before file upload
image = Image.open(r"C:\ialbYY\Programs\Monster\space (1).jpg")
st.image(image, width=900)

# Fixed sidebar section
st.sidebar.title("Advanced Menu Options")
st.sidebar.button('Touch Me')

# Capture radio selection and display it
option = st.sidebar.radio('Select', ['Data','EDA','Model Building','Toolkit','Operations','Test Center','ML'])
st.sidebar.write(f"You selected: {option}")  # Fixed syntax

# Interactive elements
with st.form("my_form"):
    num1 = st.number_input('Choose First Number', 0, 9)
    num2 = st.number_input('Choose Second Number', 0, 9)
    operation = st.selectbox('Operation', ['+', '-', '*', '/'])
    
    if st.form_submit_button('Calculate'):
        if operation == '+':
            result = num1 + num2
        elif operation == '-':
            result = num1 - num2
        elif operation == '*':
            result = num1 * num2
        elif operation == '/' and num2 != 0:
            result = num1 / num2
        else:
            result = "Error"
        
        st.success(f"Result: {result}")
        st.balloons()

# Additional UI elements
st.text_input('Email')
st.date_input('Traveling date')
st.time_input('School time')
st.text_area('Description')
st.color_picker('Choose your favorite color')