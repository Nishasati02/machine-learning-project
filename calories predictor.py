import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Load Dataset
df = pd.read_csv(r'C:\Users\Admin\OneDrive\Desktop\cpc.csv')

# Preview Dataset
print("First 5 rows of the dataset:")
print(df.head())
print("\nDataset Info:")
print(df.info())
print("\nAvailable Columns in Dataset:")
print(df.columns)

# Drop Irrelevant Columns if 'Name' exists
if 'Name' in df.columns:
    df = df.drop(columns=['Name'])
else:
    print("\n'Name' column not found in the dataset. Skipping drop operation.")

# Handle Categorical Variables
columns_to_encode = ['Gender', 'ExerciseType', 'BloodPressure(mm/hg)']
columns_to_encode = [col for col in columns_to_encode if col in df.columns]
df = pd.get_dummies(df, columns=columns_to_encode, drop_first=True)

# Simulate the Target Variable
np.random.seed(42)
df['CaloriesBurned'] = (
    df['Footsteps'] * 0.05 +
    df['Weight (KG)'] * 1.2 +
    df['Heartbeat'] * 0.3 +
    np.random.normal(0, 10, len(df))
)

# Ensure necessary features exist
required_columns = ['Footsteps', 'Weight (KG)', 'Heartbeat']
for col in required_columns:
    if col not in df.columns:
        print(f"Column '{col}' is missing. Adding dummy values.")
        df[col] = 0  # Simulate with 0s

# Define Features (X) and Target (y)
X = df.drop(columns=['CaloriesBurned'])
y = df['CaloriesBurned']

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=42)

# Model Fitting
model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# Model Evaluation
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# Output Results
print("\nModel Performance:")
print("Mean Squared Error:", mse)
print("R-Squared Value:", r2)

# Visualization
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred, color='blue', alpha=0.6)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color='red', linestyle='--', linewidth=2)
plt.xlabel("Actual Calories Burned")
plt.ylabel("Predicted Calories Burned")
plt.title("Actual vs Predicted Calories Burned")
plt.grid(True)
plt.show()

# Model Details
print("\nModel Details:")
print("Intercept:", model.intercept_)
print("Coefficients:", model.coef_)