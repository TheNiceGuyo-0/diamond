# Diamond Price Analysis Web App
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings('ignore')

# Set page config
st.set_page_config(
    page_title="Diamond Price Analysis",
    page_icon="💎",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_and_prepare_data():
    """Load and prepare the diamond dataset"""
    diamond = sns.load_dataset('diamonds')
    
    # Create derived features
    diamond['volume'] = diamond['x'] * diamond['y'] * diamond['z']
    diamond['price_per_carat'] = diamond['price'] / diamond['carat']
    diamond['surface_area'] = 2 * (diamond['x']*diamond['y'] + diamond['x']*diamond['z'] + diamond['y']*diamond['z'])
    
    return diamond

@st.cache_data
def prepare_model_data(diamond):
    """Prepare data for machine learning model"""
    diamond_model = diamond.copy()
    
    # Label encoders
    le_cut = LabelEncoder()
    le_color = LabelEncoder()
    le_clarity = LabelEncoder()
    
    diamond_model['cut_encoded'] = le_cut.fit_transform(diamond_model['cut'])
    diamond_model['color_encoded'] = le_color.fit_transform(diamond_model['color'])
    diamond_model['clarity_encoded'] = le_clarity.fit_transform(diamond_model['clarity'])
    
    # Features for model
    features = ['carat', 'cut_encoded', 'color_encoded', 'clarity_encoded', 
                'depth', 'table', 'volume', 'surface_area']
    
    X = diamond_model[features]
    y = diamond_model['price']
    
    return X, y, le_cut, le_color, le_clarity, features

@st.cache_resource
def train_model(X, y):
    """Train the Random Forest model"""
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = RandomForestRegressor(n_estimators=1000, random_state=42)
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)
    
    return model, rmse, r2, X_test, y_test, y_pred

def predict_diamond_price(model, le_cut, le_color, le_clarity, carat, cut, color, clarity, depth, table, x, y, z):
    """Predict diamond price based on characteristics"""
    try:
        cut_encoded = le_cut.transform([cut])[0]
        color_encoded = le_color.transform([color])[0]
        clarity_encoded = le_clarity.transform([clarity])[0]
        
        volume = x * y * z
        surface_area = 2 * (x*y + x*z + y*z)
        
        features_array = [[carat, cut_encoded, color_encoded, clarity_encoded, 
                          depth, table, volume, surface_area]]
        
        predicted_price = model.predict(features_array)[0]
        return predicted_price
    except Exception as e:
        st.error(f"Error in prediction: {str(e)}")
        return None

def main():
    # Header
    st.markdown('<h1 class="main-header">💎 Diamond Price Analysis & Prediction</h1>', unsafe_allow_html=True)
    
    # Load data
    with st.spinner('Loading diamond data...'):
        diamond = load_and_prepare_data()
        X, y, le_cut, le_color, le_clarity, features = prepare_model_data(diamond)
        model, rmse, r2, X_test, y_test, y_pred = train_model(X, y)
    
    # Sidebar for navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.selectbox("Choose a page:", 
                               ["Dataset Overview", "Data Visualization", "Price Prediction", "Model Performance"])
    
    if page == "Dataset Overview":
        show_dataset_overview(diamond)
    elif page == "Data Visualization":
        show_visualizations(diamond)
    elif page == "Price Prediction":
        show_price_prediction(model, le_cut, le_color, le_clarity, diamond)
    elif page == "Model Performance":
        show_model_performance(model, features, rmse, r2, X_test, y_test, y_pred)

def show_dataset_overview(diamond):
    st.header("📊 Dataset Overview")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Diamonds", f"{len(diamond):,}")
    with col2:
        st.metric("Average Price", f"${diamond['price'].mean():.2f}")
    with col3:
        st.metric("Price Range", f"${diamond['price'].min()} - ${diamond['price'].max():,}")
    with col4:
        st.metric("Average Carat", f"{diamond['carat'].mean():.2f}")
    
    st.subheader("Dataset Sample")
    st.dataframe(diamond.head(10))
    
    st.subheader("Categorical Variables")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.write("**Cut Types:**")
        st.write(diamond['cut'].value_counts())
    
    with col2:
        st.write("**Color Grades:**")
        st.write(diamond['color'].value_counts())
    
    with col3:
        st.write("**Clarity Grades:**")
        st.write(diamond['clarity'].value_counts())
    
    st.subheader("Price Statistics by Category")
    
    # Analysis by categorical variables
    cut_analysis = diamond.groupby('cut')['price'].agg(['mean', 'median', 'std', 'count'])
    color_analysis = diamond.groupby('color')['price'].agg(['mean', 'median', 'std', 'count'])
    clarity_analysis = diamond.groupby('clarity')['price'].agg(['mean', 'median', 'std', 'count'])
    
    tab1, tab2, tab3 = st.tabs(["Cut Analysis", "Color Analysis", "Clarity Analysis"])
    
    with tab1:
        st.dataframe(cut_analysis)
    with tab2:
        st.dataframe(color_analysis)
    with tab3:
        st.dataframe(clarity_analysis)

def show_visualizations(diamond):
    st.header("📈 Data Visualizations")
    
    # Price distribution
    st.subheader("Price Distribution")
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(diamond['price'], bins=50, alpha=0.7, color='skyblue', edgecolor='black')
    ax.set_title('Diamond Price Distribution')
    ax.set_xlabel('Price ($)')
    ax.set_ylabel('Frequency')
    st.pyplot(fig)
    
    # Box plots for categorical variables
    st.subheader("Price by Categorical Variables")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.boxplot(data=diamond, x='cut', y='price', ax=ax)
        ax.set_title('Price by Cut Quality')
        plt.xticks(rotation=45)
        st.pyplot(fig)
    
    with col2:
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.boxplot(data=diamond, x='color', y='price', ax=ax)
        ax.set_title('Price by Color')
        st.pyplot(fig)
    
    with col3:
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.boxplot(data=diamond, x='clarity', y='price', ax=ax)
        ax.set_title('Price by Clarity')
        plt.xticks(rotation=45)
        st.pyplot(fig)
    
    # Correlation heatmap
    st.subheader("Correlation Matrix")
    numerical_cols = ['carat', 'depth', 'table', 'price', 'x', 'y', 'z', 'volume']
    correlation_matrix = diamond[numerical_cols].corr()
    
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0, ax=ax)
    ax.set_title('Diamond Attribute Correlation Matrix')
    st.pyplot(fig)
    
    # Interactive scatter plot
    st.subheader("Interactive Analysis")
    
    # Plotly scatter plot
    fig = px.scatter(diamond.sample(5000), x='carat', y='price', color='cut', 
                    size='depth', hover_data=['color', 'clarity'],
                    title='Diamond Analysis: Carat vs Price')
    st.plotly_chart(fig, use_container_width=True)

def show_price_prediction(model, le_cut, le_color, le_clarity, diamond):
    st.header("💰 Diamond Price Prediction")
    
    st.write("Enter diamond characteristics to get a price prediction:")
    
    col1, col2 = st.columns(2)
    
    with col1:
        carat = st.number_input("Carat", min_value=0.1, max_value=5.0, value=1.0, step=0.1)
        cut = st.selectbox("Cut", diamond['cut'].unique())
        color = st.selectbox("Color", diamond['color'].unique())
        clarity = st.selectbox("Clarity", diamond['clarity'].unique())
    
    with col2:
        depth = st.number_input("Depth (%)", min_value=40.0, max_value=80.0, value=61.0, step=0.1)
        table = st.number_input("Table (%)", min_value=40.0, max_value=80.0, value=57.0, step=0.1)
        x = st.number_input("Length (mm)", min_value=1.0, max_value=15.0, value=6.5, step=0.1)
        y = st.number_input("Width (mm)", min_value=1.0, max_value=15.0, value=6.5, step=0.1)
        z = st.number_input("Height (mm)", min_value=1.0, max_value=10.0, value=4.0, step=0.1)
    
    if st.button("Predict Price", type="primary"):
        predicted_price = predict_diamond_price(model, le_cut, le_color, le_clarity, 
                                               carat, cut, color, clarity, depth, table, x, y, z)
        
        if predicted_price:
            st.success(f"🎯 Predicted Diamond Price: **${predicted_price:,.2f}**")
            
            # Show price per carat
            price_per_carat = predicted_price / carat
            st.info(f"💎 Price per Carat: **${price_per_carat:,.2f}**")
            
            # Compare with similar diamonds
            similar_diamonds = diamond[
                (diamond['cut'] == cut) & 
                (diamond['color'] == color) & 
                (diamond['clarity'] == clarity) &
                (abs(diamond['carat'] - carat) <= 0.2)
            ]
            
            if len(similar_diamonds) > 0:
                avg_similar_price = similar_diamonds['price'].mean()
                st.write(f"📊 Average price of similar diamonds: ${avg_similar_price:,.2f}")
                
                if predicted_price > avg_similar_price:
                    st.warning("This prediction is above the average for similar diamonds.")
                else:
                    st.success("This prediction is below the average for similar diamonds.")

def show_model_performance(model, features, rmse, r2, X_test, y_test, y_pred):
    st.header("🤖 Model Performance")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("RMSE", f"${rmse:,.2f}")
    with col2:
        st.metric("R² Score", f"{r2:.4f}")
    with col3:
        st.metric("Accuracy", f"{r2:.1%}")
    
    # Feature importance
    st.subheader("Feature Importance")
    feature_importance = pd.DataFrame({
        'feature': features,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(data=feature_importance, x='importance', y='feature', ax=ax)
    ax.set_title('Feature Importance in Diamond Price Prediction')
    ax.set_xlabel('Importance')
    st.pyplot(fig)
    
    # Actual vs Predicted
    st.subheader("Actual vs Predicted Prices")
    
    col1, col2 = st.columns(2)
    
    with col1:
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.scatter(y_test, y_pred, alpha=0.5)
        ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
        ax.set_xlabel('Actual Price')
        ax.set_ylabel('Predicted Price')
        ax.set_title('Actual vs Predicted Diamond Prices')
        st.pyplot(fig)
    
    with col2:
        # Residuals plot
        residuals = y_test - y_pred
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.scatter(y_pred, residuals, alpha=0.5)
        ax.axhline(y=0, color='r', linestyle='--')
        ax.set_xlabel('Predicted Price')
        ax.set_ylabel('Residuals')
        ax.set_title('Residuals Plot')
        st.pyplot(fig)
    
    # Model insights
    st.subheader("Key Insights")
    st.write(f"• The model explains **{r2:.1%}** of the variance in diamond prices")
    st.write(f"• **{feature_importance.iloc[0]['feature']}** is the most important feature")
    st.write(f"• Average prediction error is **${rmse:,.2f}**")
    st.write(f"• The model was trained on **{len(X_test) * 4:,}** diamonds")

if __name__ == "__main__":
    main()