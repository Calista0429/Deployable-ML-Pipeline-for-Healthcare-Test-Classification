from fastapi import FastAPI
import joblib
import pandas as pd
from pydantic import BaseModel, Field
from function import Pipeline  



# define model info
model_path = "./app/data/ensemble_model.pkl"
encoder_path = "./app/data/categorical_encoder.pkl"
label_encoder_path = "./app/data/label_encoder.pkl"
feature_path = "./app/data/feature_names.pkl"

# load model
model = joblib.load(model_path)
cls_encoder = joblib.load(encoder_path)
label_encoder = joblib.load(label_encoder_path)


class PatientInput(BaseModel):
    Blood_Type: str = Field(..., alias="Blood Type")
    Medical_Condition: str = Field(..., alias="Medical Condition")
    Billing_Amount: float = Field(..., alias="Billing Amount")
    Age: int = Field(..., alias="Age")
    Stay_Days: int = Field(..., alias="Stay Days")
    class Config:
        allow_population_by_field_name = True



# create FastAPI object
app = FastAPI()

# API operations
@app.get("/")
def health_check():
    return {'health_check': 'OK'}

@app.get("/info")
def info():
    return {'name': 'medical-test-pred', 'description': "Predict the medical test results"}


@app.post("/predict")
def predict(input_data: PatientInput ):


    df = pd.DataFrame([input_data.dict(by_alias=True)])
    try:
        # 预处理
        X_input = Pipeline(df, encoder=cls_encoder)
        feature_names = joblib.load(feature_path)
        X_input = X_input[feature_names]
        
        prediction = model.predict(X_input)[0]
        label = label_encoder.inverse_transform([prediction])[0]
        return {"prediction": label}
    except Exception as e:
        return {"error": f"Prediction failed: {str(e)}"}