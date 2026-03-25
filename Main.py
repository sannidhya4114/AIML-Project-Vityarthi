import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import os

# 1. Loading the saved model, columns, and the dataset
model=joblib.load('heart_model.pkl')
model_columns=joblib.load('model_columns.pkl')
df_original=pd.read_csv('heart.csv') 
print("--- Heart Disease Risk Predictor ---")

# 2. Getting User Input from Terminal
def get_user_input():
    user_data={}
    print("\nPlease enter the patient details:")
    # Loop through expected columns to ensure order matches exactly
    for col in model_columns:
        # Provide helpful descriptions for common inputs
        desc = {
            'age':'Age',
            'sex':'Sex (1=Male, 0=Female)',
            'cp':'Chest Pain Type (0-3)',
            'trestbps':'Resting Blood Pressure (mm Hg)',
            'chol':'Cholesterol (mg/dl)',
            'fbs':'Fasting Blood Sugar > 120 mg/dl (1=True, 0=False)',
            'restecg':'Resting ECG results (0-2)',
            'thalach':'Max Heart Rate Achieved',
            'exang':'Exercise Induced Angina (1=Yes, 0=No)',
            'oldpeak':'ST depression (e.g., 1.5)',
            'slope':'Slope of ST segment (0-2)',
            'ca':'Number of major vessels (0-4)',
            'thal':'Thal (0-3)'
        }.get(col,col)
        user_data[col]=float(input(f"{desc}: "))
    return pd.DataFrame([user_data])

# 3. Predict
input_df=get_user_input()
input_df=input_df[model_columns]
prediction=model.predict(input_df)[0]
probability=model.predict_proba(input_df)[0]
status = "DISEASE DETECTED" if prediction==0 else "NO DISEASE DETECTED"
# The risk probability is the chance of being in class '0'
risk_pct = probability[0] * 100 

# 4. Results & Plotting
print(f"\n"+"="*30)
print(f"RESULT: {status}")
print(f"Risk Probability: {risk_pct:.2f}%")
print("="*30+"\n")

# Plotting
plt.figure(figsize=(10, 6))
sns.scatterplot(data=df_original, x='chol', y='thalach', hue='target', alpha=0.3, palette='coolwarm')
plt.scatter(input_df['chol'], input_df['thalach'], color='black', s=300, marker='*', label='Current Patient')
plt.title(f"Patient Analysis Result: {status}")
plt.xlabel("Cholesterol (mg/dl)")
plt.ylabel("Max Heart Rate (thalach)")
plt.legend()
plt.grid(True, linestyle='--', alpha=0.6)
plt.show()