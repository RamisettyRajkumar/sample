import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Load the dataset
file_path = 'weather.csv'  # Ensure this is the correct path to your dataset
df = pd.read_csv(file_path)

# Data Exploration
print("Data Information:\n")
print(df.info())
print("\nData Description:\n")
print(df.describe())

# Data Visualization - Pair plot
sns.pairplot(df)
plt.title('Pair Plot of Weather Data')
plt.show()

# Feature Engineering (if needed)
# Example: Create a new feature for temperature range
if 'MinTemp' in df.columns and 'MaxTemp' in df.columns:
    df['TempRange'] = df['MaxTemp'] - df['MinTemp']

# Data Analysis
# Example: Calculate the correlation matrix
correlation_matrix = df.corr()
print("\nCorrelation Matrix:\n")
print(correlation_matrix)

# Data Visualization (Part 2) - Heatmap of the correlation matrix
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', linewidths=0.5)
plt.title('Correlation Matrix Heatmap')
plt.show()

# Advanced Analysis - Rainfall prediction using Linear Regression
# Ensure the dataset has the necessary columns for this analysis
if 'Rainfall' in df.columns and 'MinTemp' in df.columns and 'MaxTemp' in df.columns:
    features = df[['MinTemp', 'MaxTemp']]
    target = df['Rainfall']
    
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)
    
    # Train a linear regression model
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    # Make predictions on the test set
    y_pred = model.predict(X_test)
    
    # Evaluate the model
    mse = mean_squared_error(y_test, y_pred)
    print("\nLinear Regression Model - Mean Squared Error:", mse)

# Conclusions and Insights
print("\nConclusions and Insights:")
print("1. Pair plot and heatmap visualizations reveal relationships between variables.")
print("2. Linear regression model shows a Mean Squared Error of", mse, "for rainfall prediction based on MinTemp and MaxTemp.")

# Save or Display Results
# The script provides results through the terminal or command prompt.
# Additional results can be exported as needed.
