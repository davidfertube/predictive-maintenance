import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from huggingface_hub import InferenceClient
import os

# Constants
HF_TOKEN = os.environ.get("HF_TOKEN")
MODEL_ID = "mistralai/Mistral-7B-Instruct-v0.3"

class MaintenanceEngine:
    def __init__(self):
        self.client = InferenceClient(model=MODEL_ID, token=HF_TOKEN)

    def predict_rul(self, sensor_history):
        # Simplified RUL prediction using linear trend of degradation
        # In a senior app, this would be a pre-trained LSTM or XGBoost
        df = pd.DataFrame(sensor_history)
        X = np.arange(len(df)).reshape(-1, 1)
        y = df["health_index"].values
        
        reg = LinearRegression().fit(X, y)
        slope = reg.coef_[0]
        intercept = reg.intercept_
        
        # Calculate when health_index hits 0
        if slope >= 0:
            return 999 # Not failing
        
        rul = -intercept / slope - len(df)
        return max(0, int(rul))

    def get_maintenance_strategy(self, asset_id, rul):
        prompt = f"""You are a Senior Maintenance Planner. 
Asset {asset_id} has an estimated Remaining Useful Life (RUL) of {rul} cycles. 
Provide a high-level maintenance strategy and spare parts procurement plan."""

        response = self.client.chat_completion(
            messages=[{"role": "user", "content": prompt}],
            max_tokens=600
        )
        return response.choices[0].message.content

maintenance_engine = MaintenanceEngine()
