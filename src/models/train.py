import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
import pickle
import shap
import os
import matplotlib.pyplot as plt
import yaml

# Import params.yaml and define variables
params = yaml.safe_load(open('params.yaml', 'r'))['train_model']
test_size = params['test_size']
n_estimators = params['n_estimators']
max_depth = params['max_depth']
learning_rate = params['learning_rate']
model_type = params['model']


def get_feature_names(trf, original_feature_names):
    ohe_feature_names = trf.named_transformers_['team_ohe'].get_feature_names_out(original_feature_names[:2])
    remainder_feature_names = original_feature_names[2:]
    return np.concatenate([ohe_feature_names, remainder_feature_names])

def preprocessing(X_train, y_train, X_test, y_test):
    # Define the ColumnTransformer correctly
    trf1 = ColumnTransformer([
        ('team_ohe', OneHotEncoder(handle_unknown='ignore', drop='first'), [0, 1])
    ], remainder='passthrough')
    
    if model_type == 'xgboost':
        model = XGBRegressor(n_estimators=n_estimators, max_depth=max_depth, learning_rate=learning_rate)
    elif model_type == 'rfreg':
        model = RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth)
    else:
        raise ValueError(f"Unsupported model type: {model_type}")
    
    pipe = Pipeline([
        ('trf1', trf1),
        ('model', model)
    ])
    pipe.fit(X_train, y_train)
    
    y_pred = pipe.predict(X_test.iloc[:1, :])
    y_true = y_test.iloc[:1].values.ravel()
    rmse = mean_squared_error(y_true, y_pred, squared=False)  # RMSE calculation

    # Get feature names after transformation
    feature_names = get_feature_names(trf1, original_feature_names)

    # SHAP explainability
    # Convert the transformed data to a dense format
    X_train_transformed = pipe.named_steps['trf1'].transform(X_train).toarray()
    X_test_transformed = pipe.named_steps['trf1'].transform(X_test.iloc[:1, :]).toarray()

    explainer = shap.Explainer(pipe.named_steps['model'], X_train_transformed)
    shap_values = explainer(X_test_transformed)

    # Ensure the 'models' directory exists
    os.makedirs('models', exist_ok=True)

    # Save SHAP values for further analysis
    np.save('models/shap_values.npy', shap_values.values)
    #shap.summary_plot(shap_values, X_test_transformed, show=False)
    shap.waterfall_plot(shap.Explanation(values=shap_values[0].values, base_values=shap_values[0].base_values, data=X_test_transformed[0], feature_names=feature_names), show=False)
    #plt.savefig('models/shap_summary_plot.png')
    plt.savefig('models/shap_waterfall.png')
    
    return pipe, rmse,y_pred

df_final = pd.read_csv("./data/processed/df_final.csv")
X = df_final.drop(['total_score'], axis=1)
y = df_final['total_score']
original_feature_names = X.columns.tolist()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=22)

pipe, rmse,y_pred= preprocessing(X_train, y_train, X_test, y_test)
print(rmse)
print(y_pred)
print(y_test.iloc[:1])
print(X_test.iloc[:1])


# Ensure the 'models' directory exists
os.makedirs('models', exist_ok=True)

# Save the model to the 'models' directory
pickle.dump(pipe, open('models/pipe.pkl', 'wb'))
