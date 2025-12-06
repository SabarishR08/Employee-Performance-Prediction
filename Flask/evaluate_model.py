import pickle
import pandas as pd
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import numpy as np

# Paths
MODEL_FILE = r"E:\Downloads - D\MLE\Flask\gwp.pkl"
TEST_FILE = r"E:\Downloads - D\MLE\Dataset\garments_worker_productivity.csv"

# Load model
with open(MODEL_FILE, 'rb') as f:
    model = pickle.load(f)
print("✅ Model loaded successfully.")

# Load dataset
df = pd.read_csv(TEST_FILE)
print(f"✅ Dataset loaded. Shape: {df.shape}")

# Features expected by the model
features = [
    'quarter', 'department', 'day', 'team', 'targeted_productivity', 'smv',
    'over_time', 'incentive', 'idle_time', 'idle_men',
    'no_of_style_change', 'no_of_workers', 'month'
]

# Ensure all features exist
for col in features:
    if col not in df.columns:
        df[col] = 0

X_test = df[features].apply(pd.to_numeric, errors='coerce').fillna(0)
y_test = df['actual_productivity'].apply(pd.to_numeric, errors='coerce').fillna(0)

# Predict
y_pred = model.predict(X_test)

# Metrics
r2 = r2_score(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
mae = mean_absolute_error(y_test, y_pred)

print("\n📊 Model Performance on Test Data:")
print(f"R² Score      : {r2:.4f}")
print(f"RMSE          : {rmse:.4f}")
print(f"MAE           : {mae:.4f}")

# Optional: Show first 10 predictions vs actual
comparison = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
print("\nFirst 10 predictions vs actual:")
print(comparison.head(10))
