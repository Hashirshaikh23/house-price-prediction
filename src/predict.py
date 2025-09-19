# src/predict.py
import joblib
import pandas as pd

def load_model(model_path):
    data = joblib.load(model_path)
    return data

def predict_single(model_data, row_dict):
    # expects row_dict keys match model_data['columns']
    df = pd.DataFrame([row_dict], columns=model_data['columns'])
    # impute & predict via pipeline stored in model
    model = model_data['model']
    pred = model.predict(df)
    return float(pred[0])

if __name__ == "__main__":
    model_data = load_model("../models/best_model.pkl")
    sample = {c: 0 for c in model_data['columns']}
    # fill a few with typical values, e.g. RM:
    sample['RM'] = 6.5
    print("Prediction:", predict_single(model_data, sample))
