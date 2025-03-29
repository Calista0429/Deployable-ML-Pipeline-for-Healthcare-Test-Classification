import pandas as pd
from sklearn.preprocessing import StandardScaler
def Pipeline(input_df: pd.DataFrame, encoder: dict):
    
    numerical_cols = ['Billing Amount','Age','Stay Days']
    categorical_cols = ['Blood Type', 'Medical Condition']

    categorical_feature = pd.DataFrame()

    for col in categorical_cols:
        le = encoder[col]
        categorical_feature[col] = le.transform(input_df[col].astype(str))

    numerical_feature = pd.DataFrame(input_df[numerical_cols], columns=numerical_cols)

    feature = pd.concat([categorical_feature,numerical_feature], axis=1)
    return feature

    