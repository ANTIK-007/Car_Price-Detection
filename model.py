import pandas as pd
import pickle
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# Load dataset
df = pd.read_csv("CarPrice_Assignment.csv")
df = df.drop("symboling", axis=1)

# Encode categorical variables
le = LabelEncoder()
categorical_cols = [
    "CarName", "fueltype", "aspiration", "doornumber",
    "carbody", "drivewheel", "enginelocation",
    "enginetype", "cylindernumber", "fuelsystem"
]

for col in categorical_cols:
    df[col] = le.fit_transform(df[col])

X = df.iloc[:, :24]
y = df.iloc[:, 24]

# Scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train model
model = LinearRegression()
model.fit(X_scaled, y)

# Save model & scaler
pickle.dump(model, open("model.pkl", "wb"))
pickle.dump(scaler, open("scaler.pkl", "wb"))

print("Model and scaler saved successfully.")
