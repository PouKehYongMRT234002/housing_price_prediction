import pandas as pd
import joblib
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LinearRegression

# Load dataset
df = pd.read_csv("Housing.csv")

# Define features
num_features = ["area", "bedrooms", "bathrooms", "stories", "parking"]
cat_features = [
    "mainroad", "guestroom", "basement",
    "hotwaterheating", "airconditioning",
    "prefarea", "furnishingstatus"
]

X = df[num_features + cat_features]
y = df["price"]

# Preprocessing
preprocess = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat_features),
        ("num", "passthrough", num_features),
    ]
)

# Pipeline
model_v2 = Pipeline([
    ("preprocess", preprocess),
    ("regressor", LinearRegression())
])

# Train
model_v2.fit(X, y)

# Save model
joblib.dump(model_v2, "model_v2.pkl")
print("Improved model v2 saved correctly.")
