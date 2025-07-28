#imports for data handling
import pandas as pd #reading and manipulationg data
import numpy as np #for numerical operations

#import for data visualization
import matplotlib.pyplot as plt #basic ploting
import seaborn as sns #statistical data visualiztion
import plotly.express as px
import plotly.graph_objects as go

#for preprocessing and model training
from sklearn.model_selection import train_test_split #encode categories
from sklearn.ensemble import RandomForestRegressor #split data for model training
from sklearn.metrics import mean_squared_error, r2_score #model evaluation
from sklearn.preprocessing import LabelEncoder
#for warnings
import warnings
warnings.filterwarnings('ignore')

diamond = sns.load_dataset('diamonds')
#diamond['price'] = diamond['price'] * 100

#diamond = pd.read_csv('diamonds.csv') #pd.read_csv; reads the data on the dataset
# print(diamond.shape) #shows numbers of rows and columns
# print(diamond.isnull().sum()) #finds missing data
# print(diamond.columns)
# print(diamond.dtypes)

#labeling unique categorial values
print('Categorial Values:')
for col in ['cut', 'color', 'clarity']:
    print(f'{col}: {diamond[col].unique()}') #get the unique categorical values

#print(f'Price statistics:\n{diamond['price'].describe}')

#histogram of prices
plt.figure(figsize=(12, 6)) #get figure size
plt.hist(diamond['price'], bins=50, alpha=0.7) #histogram of the price row with 50 intervals
plt.title('Diamond price') #chart title
plt.xlabel('Price') #the x-axis
plt.ylabel('Frequency') #the y-axis

#price by cut
plt.subplot(2, 2, 2) #divides plot area into 2 rows and 2 columns and selects 2nd subplot (top-right)
sns.boxplot(data=diamond, x='cut', y='price') #compare color vs price
plt.title('Price by Cut Quality')
plt.xticks(rotation=45) #better readability

#price by color
plt.subplot(2, 2, 3)
sns.boxplot(data=diamond, x='color', y='price')
plt.title('Price by Color')

#price byclarity
plt.subplot(2, 2, 4) 
sns.boxplot(data=diamond, x='clarity', y='price') 
plt.title('Price by clarity')
plt.xticks(rotation=45)

plt.subplots_adjust(hspace=1.0, wspace=0.9)#used to manually add spacing to prevent overlapping labels, titles or axes in multi-plot figures
plt.show

#correlation heatmap
plt.figure(figsize=(10, 8))
numerical_cols = ['carat', 'depth', 'table', 'price', 'x', 'y', 'z'] #label all numerical features
correlation_matrix = diamond[numerical_cols].corr() #.corr() is used to assesses the strength and direction of a linear relationship between two variables
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0) #visualises the values
plt.title('Diamond Attribute Correlation Matrix')
plt.show()

#scatter plot for key relationships
fig, axes = plt.subplots(2, 2, figsize=(15, 10))


#carat vs prices
axes[0, 0].scatter(diamond['carat'], diamond['price'], alpha=0.5)
axes[0, 0].set_xlabel('Carat')
axes[0, 0].set_ylabel('Price')
axes[0, 0].set_title('Carat vs Price')

#depth vs price
axes[0, 1].scatter(diamond['depth'], diamond['price'], alpha=0.5)
axes[0, 1].set_xlabel('Depth')
axes[0, 1].set_ylabel('Price')
axes[0, 1].set_title('Depth vs Price')

#table vs price
axes[1, 0].scatter(diamond['table'], diamond['price'], alpha=0.5)
axes[1, 0].set_xlabel('Table')
axes[1, 0].set_ylabel('Price')
axes[1, 0].set_title('Table vs Price')

#diamon volume (x*y*z) vs price
diamond['volume'] = diamond['x'] * diamond['y'] * diamond['z']
axes[1, 1].scatter(diamond['volume'], diamond['price'], alpha=0.5)
axes[1, 1].set_xlabel('Volume')
axes[1, 1].set_ylabel('Price')
axes[1, 1].set_title('Volume vs Price')

plt.tight_layout(pad=4.0)
plt.show


#analysis by categorical variables
cut_analysis = diamond.groupby('cut')['price'].agg(['mean', 'median', 'std', 'count']) #groups data by cut and aggregates price values for each cut to compute mean, median, std(standard deviation - price variabilty) and count
color_analysis = diamond.groupby('color')['price'].agg(['mean', 'median', 'std', 'count']) #groups data by cut and aggregates price values for each cut to compute mean, median, std(standard deviation - price variabilty) and count
clarity_analysis = diamond.groupby('clarity')['price'].agg(['mean', 'median', 'std', 'count']) #groups data by cut and aggregates price values for each cut to compute mean, median, std(standard deviation - price variabilty) and count

print('Cut Analysis')
print(cut_analysis)
print('\n Color Analysis')
print(color_analysis)
print('\n Clarity Analysis')
print(clarity_analysis)

#analysis for price per carat
diamond['price_per_carat'] = diamond['price'] / diamond['carat']
print(f'\n Price per Carat Stat:\n {diamond['price_per_carat'].describe}')

diamond_model = diamond.copy() #create a copy of the database for the model

dm_cut = LabelEncoder()#making categorical variables readable for the model
dm_color = LabelEncoder()
dm_clarity = LabelEncoder()

diamond_model['cut_encoded'] = dm_cut.fit_transform(diamond_model['cut'])
diamond_model['color_encoded'] = dm_color.fit_transform(diamond_model['color'])
diamond_model['clarity_encoded'] = dm_clarity.fit_transform(diamond_model['clarity'])

#making the volume and surface area feature
diamond_model['volume'] = diamond_model['x'] * diamond_model['y'] * diamond_model['z']
diamond_model['surface_area'] = 2 * (diamond_model['x']*diamond_model['y'] + diamond_model['x']*diamond_model['z'] + diamond_model['y']*diamond_model['z']) #uses the formula 2*(xy + xz + yz) to find surface area of a cuboid since we dont have actual cut shapes

#pick the features for the model
features = ['carat', 'cut_encoded', 'color_encoded', 'clarity_encoded', 
            'depth', 'table', 'volume', 'surface_area']

X = diamond_model[features]
y = diamond_model['price']

#modeling [predictive modelling]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#train random forest model: this model combimnes multiple decision trees to make predictions, it is mostly used for classification and regresstion(mean prediction) task cause it can handle complex data sets
forest_model = RandomForestRegressor(n_estimators=1000, random_state=42)
forest_model.fit(X_train, y_train)

#make predictions based on x
y_pred = forest_model.predict(X_test)

#evaluate the model
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print(f"Model Performance:")
print(f"RMSE: ${rmse:.2f}")
print(f"R² Score: {r2:.4f}")

#feature importance: checking which feature contributed the most to the random forest prediction
feature_importance = pd.DataFrame({
    'feature': features,
    'importance': forest_model.feature_importances_
}).sort_values('importance', ascending=False)

print(f'\nFeature Importance:')
print(feature_importance)

#Advance Visualization
# Feature importance plot
plt.figure(figsize=(10, 6))
sns.barplot(data=feature_importance, x='importance', y='feature')
plt.title('Feature Importance in Diamond Price Prediction')
plt.xlabel('Importance')
plt.show()

# Actual vs Predicted prices
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, alpha=0.5)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
plt.xlabel('Actual Price')
plt.ylabel('Predicted Price')
plt.title('Actual vs Predicted Diamond Prices')
plt.show()

# Residuals plot
residuals = y_test - y_pred
plt.figure(figsize=(10, 6))
plt.scatter(y_pred, residuals, alpha=0.5)
plt.axhline(y=0, color='r', linestyle='--')
plt.xlabel('Predicted Price')
plt.ylabel('Residuals')
plt.title('Residuals Plot')
plt.show()

#interactive Analysis
# Interactive scatter plot
# fig = px.scatter(diamond, x='carat', y='price', color='cut', 
#                 size='depth', hover_data=['color', 'clarity'],
#                 title='Interactive Diamond Analysis: Carat vs Price')
# fig.show()

# # 3D scatter plot
# fig = px.scatter_3d(diamond.sample(1000), x='carat', y='depth', z='price',
#                    color='cut', title='3D Diamond Analysis')
# fig.show()

#price prediction function: predicts price based on characteristics
def predict_diamond_price(carat, cut, color, clarity, depth, table, x, y, z):
    #encode categorial variables
    cut_encoded = dm_cut.transform([cut])[0]
    color_encoded = dm_color.transform([color])[0]
    clarity_encoded = dm_clarity.transform([clarity])[0]

    #calc derived features
    volume = x * y * z
    surface_area = 2 * (x*y + x*z + y*z)

    #create features array
    features_array = [[carat, cut_encoded, color_encoded, clarity_encoded, 
                      depth, table, volume, surface_area]]
    
    #predict price
    predicted_price = forest_model.predict(features_array)[0]

    return predicted_price


def generate_insights_report():
    """
    Generate key insights from the analysis
    """
    insights = {
        'dataset_size': len(diamond),
        'avg_price': diamond['price'].mean(),
        'most_common_cut': diamond['cut'].mode()[0],
        'highest_correlation_with_price': correlation_matrix['price'].abs().sort_values(ascending=False).index[1],
        'price_range': f"${diamond['price'].min()} - ${diamond['price'].max()}",
        'model_accuracy': f"{r2:.2%}",
        'most_important_feature': feature_importance.iloc[0]['feature']
    }
    
    print("=== DIAMOND PRICE ANALYSIS INSIGHTS ===")
    for key, value in insights.items():
        print(f"{key.replace('_', ' ').title()}: {value}")
    
    return insights

# Generate report
report = generate_insights_report()

example_price = predict_diamond_price(
    carat=1.0, cut='Ideal', color='E', clarity='VS1',
    depth=61.0, table=57.0, x=6.5, y=6.5, z=4.0
)
print(f"Predicted price for example diamond: ${example_price:.2f}")

