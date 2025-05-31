# air_index_app.py
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import gdown
import os

# ------------------- PAGE CONFIG ------------------- #
st.set_page_config(
    page_title="Air Index App",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ------------------- LOAD DATA ------------------- #
@st.cache_data

def load_data():
    file_path = "cleaned_data.csv"
    file_id = "1nnM-U_h1ufa8yafaIsciMr6x3t-2W5X3"
    url = f"https://drive.google.com/uc?id={file_id}"
    st.write("📥 Checking for dataset...")
    output = 'cleaned_data.csv'
    gdown.download(url, file_path, quiet=False)
    df = pd.read_csv(file_path)


    if not os.path.exists(file_path):
        st.info("Downloading dataset from Google Drive...")
        try:
            output = gdown.download(url, file_path, quiet=False)
            if output is None:
                st.error("❌ gdown.download returned None. Possibly wrong file ID or access issue.")
                return None
        except Exception as e:
            st.error(f"❌ Exception during download: {e}")
            return None
    

    try:
        
        data = pd.read_csv(file_path)
        numeric_cols = data.select_dtypes(include=['number']).columns
        data[numeric_cols] = data[numeric_cols].fillna(data[numeric_cols].mean())
        st.success("✅ Data loaded successfully.")
        return data
    except Exception as e:
        st.error(f"❌ Failed to read CSV: {e}")
        return None
    
    
data = load_data()

# ------------------- SIDEBAR ------------------- #
st.sidebar.title("Navigation")
page = st.sidebar.selectbox("Go to", ["Welcome", "Data Overview", "EDA Analysis", "Model Prediction"])

# ------------------- WELCOME PAGE ------------------- #
if page == "Welcome":
    st.title("🌍 Air Index App")
    try:
        image = Image.open("space.jpg")
        st.image(image, width=900)
    except:
        st.warning("Header image not found.")

    st.markdown("""
        ### Welcome to the Air Quality Forecast App

        This platform helps you explore air quality data and predict pollution levels using machine learning techniques.

        **Sections:**
        - 📁 View raw dataset overview
        - 📊 Perform Exploratory Data Analysis (EDA)
        - 🤖 Build prediction models for PM2.5
    """)

# ------------------- DATA OVERVIEW ------------------- #
elif page == "Data Overview":
    st.title("📁 Dataset Overview")
    if data is not None:
        st.success("Dataset loaded successfully!")
        st.write(f"\U0001F4CF Rows: {len(data)} | Columns: {len(data.columns)}")
        st.write(f"🔍 Missing Values: {data.isna().sum().sum()}")
        preview_type = st.radio("Choose view", ["Head", "Tail", "Random"], horizontal=True)
        if preview_type == "Head":
            st.dataframe(data.head())
        elif preview_type == "Tail":
            st.dataframe(data.tail())
        else:
            st.dataframe(data.sample(30))
    else:
        st.error("No data available.")

# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 


# ------------------- EDA ANALYSIS ------------------- #
elif page == "EDA Analysis":
    st.title("📊 Exploratory Data Analysis")

    st.subheader("📊 Dataset Summary")
    st.markdown("""
    This dataset includes **hourly records from March 1, 2013 to February 28, 2017**,
    collected at **12 national air-quality monitoring stations** across Beijing:

    - **Pollutants Tracked:** PM2.5, PM10, SO₂, NO₂, CO, O₃
    - **Meteorological Variables:** Wind Speed (WSPM), Rainfall (RAIN), 
    Temperature (TEMP), Dew Point (DEWP), Pressure (PRES)
    - **Sources:**
        - Air-quality: *Beijing Municipal Environmental Monitoring Center*
        - Weather: *China Meteorological Administration*
    """)

    # Station Classification
    st.subheader("📍 Station Classification")
    st.markdown("""
    To ensure **comprehensive data analysis**, stations are categorized by their surrounding environment:

    | Environment Type       | Station Name      | Description |
    |------------------------|-------------------|-------------|
    | **Urban**              | **Dongsi**        | Located in central Beijing, high population density and traffic. |
    | **Suburban**           | **Shunyi**        | Outskirts of Beijing, moderate development, residential. |
    | **Rural**              | **Huairou**       | Mountainous and forested, low population density. |
    | **Industrial/Hotspot** | **Nongzhanguan**  | Commercial zone, near major roads and industrial activity. |

    💡 You may select and analyze **one station from each type** to compare urbanization and pollution trends.
    """)

    if data is not None:
        
        # Sidebar filter
        st.sidebar.header("🔎 Filters")
        selected_station = st.sidebar.selectbox("Select Monitoring Station", data['station'].unique())
        filtered_data = data[data['station'] == selected_station]

        st.success(f"Showing analysis for station: **{selected_station}**")

        st.subheader("📌 Statistics of Numeric Features")
        st.dataframe(filtered_data.describe())

        # Columns for layout
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("📉 Missing Values")
            st.bar_chart(filtered_data.isna().sum())

        with col2:
            st.subheader("📊 Pollutant Counts")
            pollutants = ['PM2.5', 'PM10', 'SO2', 'NO2', 'CO', 'O3']
            st.bar_chart(filtered_data[pollutants].count())
            
        pollutants = ['PM2.5', 'PM10', 'SO2', 'NO2', 'CO', 'O3']
        pollutants_df = pd.DataFrame({
            'Pollutant': pollutants,
            'Level': data[pollutants].mean().values
        })

    

        st.subheader("🧪 Key Observations and Insights from the Beijing Air Quality Analysis")

        st.markdown("""
        ### 📌 1. Data Overview
        The dataset includes hourly air quality readings from **12 monitoring stations** in Beijing.

        - **Date Range:** March 1, 2013 – February 28, 2017  
        - **Pollutants Tracked:** PM2.5, PM10, SO₂, NO₂, CO, O₃  
        - **Meteorological Data:** TEMP, DEWP, RAIN, WSPM, PRES  

        ---

        ### 📉 2. Missing Data
        - Significant missing values visualized via *missingno* plots.  
        - **PM2.5 and PM10** columns showed the highest number of missing values, indicating potential sensor failures or reporting issues.

        ---

        ### 📊 3. Pollutant Distribution
        - PM2.5 and PM10 have **long right-tailed distributions** → high pollution spikes.  
        - Some pollutants show **seasonal variation**, especially PM2.5 being higher in **winter months** due to heating and weather conditions.

        ---

        ### 🌍 4. Temporal Trends
        **Yearly Analysis:**
        - 2013 and 2014 show **higher PM2.5 levels**, with a gradual improvement by 2016.

        **Monthly Trends:**
        - Pollution levels **peak during December–February**, likely due to increased heating and stagnant air.

        **Hourly Patterns:**
        - PM2.5 levels rise in the **early morning and evening**, aligning with **rush hours**.

        ---

        ### 📍 5. Correlation Analysis
        - **Strong positive correlation** between PM2.5 and PM10, indicating shared sources.  
        - **Weak or negative correlations** between PM2.5 and temperature/rainfall, suggesting weather influences pollution dissipation.

        ---

        ### ⚙️ 6. Modeling (Random Forest Regressor)
        A regression model was built to **predict PM2.5 levels** using weather and other pollutants.

        **Model Evaluation:**
        - **R² Score:** ~0.78 (good)  
        - **RMSE:** Acceptable for hourly air quality prediction

        **Top features influencing PM2.5:**
        - PM10  
        - NO₂  
        - Temperature  
        - Wind Speed 
        """)
        
        
         # --- Pollutant Pie Chart ---
        st.header("🥧 Pollutant Contribution Pie Chart")
        pollutants_df = pd.DataFrame({
        'Pollutant': pollutants,
        'Level': data[pollutants].mean().values
    })
        
        
         # --- PM2.5 Distribution ---
        st.header("🌫️ PM2.5 Distribution Histogram")
        fig1, ax1 = plt.subplots(figsize=(10, 6))
        sns.histplot(data['PM2.5'].dropna(), kde=True, ax=ax1, color="skyblue", bins=30)
        ax1.set_title("Histogram of PM2.5")
        st.pyplot(fig1)

        # --- Feature Correlation Heatmap ---
        st.header("🔗 Feature Correlation Heatmap")
        numeric_data = data.select_dtypes(include=['float64', 'int64'])
        if numeric_data.empty:
            st.warning("No numeric columns available for correlation.")
    else:
        fig2, ax2 = plt.subplots(figsize=(12, 8))
        sns.heatmap(numeric_data.corr(), annot=True, cmap='coolwarm', ax=ax2, fmt='.2f', linewidths=0.5)
        ax2.set_title("Correlation Matrix of Features")
        st.pyplot(fig2)

        
         # --- PM2.5 vs Temperature Scatter ---
    st.header("🔥 PM2.5 vs Temperature")
    fig3, ax3 = plt.subplots(figsize=(10, 6))
    sns.scatterplot(data=data, x='TEMP', y='PM2.5', ax=ax3, color="orange")
    ax3.set_title("Scatter Plot: PM2.5 vs Temperature")
    st.pyplot(fig3)

    # --- PM2.5 vs PM10 Box Plot ---
    st.header("📦 PM2.5 vs PM10 Box Plot")
    filtered_data = data[['PM10', 'PM2.5']].dropna()
    if not filtered_data.empty:
        fig4, ax4 = plt.subplots(figsize=(10, 6))
        sns.boxplot(data=filtered_data, x='PM10', y='PM2.5', ax=ax4, palette="Set2")
        ax4.set_title("Box Plot of PM2.5 vs PM10")
        st.pyplot(fig4)
    else:
        st.warning("Insufficient data for Box Plot.")

    

    fig5, ax5 = plt.subplots(figsize=(8, 8))
    explode = [0.1] + [0] * (len(pollutants_df) - 1)
    ax5.pie(pollutants_df['Level'], explode=explode, labels=pollutants_df['Pollutant'],
            autopct='%1.1f%%', shadow=True, startangle=140)
    ax5.set_title("Average Pollutant Contribution")
    st.pyplot(fig5)




# ------------------- MODEL PREDICTION ------------------- 
# ------------------- PAGE ROUTING ------------------- #
if page == "EDA Analysis":
    # Your EDA analysis code here
    pass  # Replace with actual EDA code

elif page == "Model Prediction":
    st.title("🤖 Model Building")

    if data is not None:
        # Display missing values
        st.subheader("📉 Missing Values")
        st.bar_chart(data.isna().sum())

        # PM2.5 Distribution
        st.subheader("🌡️ PM2.5 Distribution")
        fig1, ax1 = plt.subplots()
        sns.histplot(data['PM2.5'].dropna(), kde=True, bins=30, ax=ax1, color='skyblue')
        st.pyplot(fig1)

        # Correlation Heatmap
        st.subheader("📊 Correlation Heatmap")
        fig2, ax2 = plt.subplots(figsize=(10, 6))
        numeric_data = data.select_dtypes(include=['number'])
        sns.heatmap(numeric_data.corr(), annot=True, cmap='coolwarm', fmt=".2f", ax=ax2)

        st.pyplot(fig2)

        # Model training and evaluation
        features = ['PM10', 'SO2', 'NO2', 'CO', 'O3', 'TEMP', 'PRES', 'DEWP', 'RAIN', 'WSPM']
        target = 'PM2.5'

        X = data[features].dropna()
        y = data.loc[X.index, target]

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

        model_choice = st.selectbox("Choose a model", ["Linear Regression", "Random Forest(Time Consuming)", "Gradient Boosting"])

        if model_choice == "Linear Regression":
            model = LinearRegression()
        elif model_choice == "Random Forest":
            model = RandomForestRegressor(n_estimators=100, random_state=42)
        else:
            model = GradientBoostingRegressor(random_state=42)

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        # Display results (you'd run model prediction before this part)
        st.markdown("---")
        st.markdown("**Mean Squared Error**")
        st.markdown(f"<h1 style='font-size: 48px; color: #333;'>{mse:.2f}</h1>", unsafe_allow_html=True)

        st.markdown("**R-squared Score**")
        st.markdown(f"<h1 style='font-size: 48px; color: #333;'>{r2:.2f}</h1>", unsafe_allow_html=True)
        st.write(f"**MSE:** {mean_squared_error(y_test, y_pred):.2f}")
        st.write(f"**R² Score:** {r2_score(y_test, y_pred):.2f}")

        # Prediction visualization
        fig, ax = plt.subplots()
        sns.scatterplot(x=y_test, y=y_pred, ax=ax)
        ax.set_xlabel("Actual PM2.5")
        ax.set_ylabel("Predicted PM2.5")
        ax.set_title(f"{model_choice} - Actual vs Predicted")
        st.pyplot(fig)

    else:
        st.warning("No data available for modeling.")
