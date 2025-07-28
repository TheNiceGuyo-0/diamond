# diamond_price_app.py

import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# Load dataset
@st.cache_data
def load_data():
    return sns.load_dataset('diamonds')

diamond = load_data()

# Encode categorical variables
dm_cut = LabelEncoder()
dm_color = LabelEncoder()
dm_clarity = LabelEncoder()

diamond['cut_encoded'] = dm_cut.fit_transform(diamond['cut'])
diamond['color_encoded'] = dm_color.fit_transform(diamond['color'])
diamond['clarity_encoded'] = dm_clarity.fit_transform(diamond['clarity'])
diamond['volume'] = diamond['x'] * diamond['y'] * diamond['z']
diamond['surface_area'] = 2 * (diamond['x']*diamond['y'] + diamond['x']*diamond['z'] + diamond['y']*diamond['z'])

# Modeling
features = ['carat', 'cut_encoded', 'color_encoded', 'clarity_encoded', 
            'depth', 'table', 'volume', 'surface_area']
X = diamond[features]
y = diamond['price']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
forest_model = RandomForestRegressor(n_estimators=1000, random_state=42)
forest_model.fit(X_train, y_train)

# Streamlit UI
st.title("💷 Diamond Price Predictor")

st.sidebar.header("Diamond Characteristics")
carat = st.sidebar.slider("Carat", 0.2, 5.0, 1.0)
cut = st.sidebar.selectbox("Cut", diamond['cut'].unique())
color = st.sidebar.selectbox("Color", diamond['color'].unique())
clarity = st.sidebar.selectbox("Clarity", diamond['clarity'].unique())
depth = st.sidebar.slider("Depth", float(diamond['depth'].min()), float(diamond['depth'].max()), 61.0)
table = st.sidebar.slider("Table", float(diamond['table'].min()), float(diamond['table'].max()), 57.0)
x = st.sidebar.slider("X (length)", 0.0, 10.0, 6.5)
y_val = st.sidebar.slider("Y (width)", 0.0, 10.0, 6.5)
z = st.sidebar.slider("Z (depth)", 0.0, 10.0, 4.0)

# Prediction
cut_encoded = dm_cut.transform([cut])[0]
color_encoded = dm_color.transform([color])[0]
clarity_encoded = dm_clarity.transform([clarity])[0]
volume = x * y_val * z
surface_area = 2 * (x*y_val + x*z + y_val*z)

features_array = [[carat, cut_encoded, color_encoded, clarity_encoded, 
                   depth, table, volume, surface_area]]
predicted_price = forest_model.predict(features_array)[0]

st.subheader("💰 Predicted Price")
st.success(f"${predicted_price:,.2f}")

# Optional - Visuals
if st.checkbox("Show Correlation Heatmap"):
    st.subheader("🔍 Price Correlation Heatmap")
    numerical_cols = ['carat', 'depth', 'table', 'price', 'x', 'y', 'z']
    correlation_matrix = diamond[numerical_cols].corr()
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', ax=ax)
    st.pyplot(fig)
