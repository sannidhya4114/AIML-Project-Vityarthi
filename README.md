# AIML-Project-Vityarthi
# Heart Disease Risk Predictor (ML)
A Machine Learning-based diagnostic tool that predicts the probability of heart disease using clinical patient metrics. 
This project uses a Random Forest Classifier to provide real-time risk assessments through a command-line interface.

# Overview
The system processes 13 clinical features—including cholesterol levels, heart rate, and ST-segment depression—to determine whether a patient shows signs of cardiovascular disease. 
It provides both a binary classification (Detected/Not Detected) and a specific risk percentage.

# Project Structure
1. Main.py: The entry point script for running predictions and generating visualizations.

2. heart_model.pkl: The serialized, pre-trained Random Forest model.

3. model_columns.pkl: Metadata file ensuring input features remain in the correct order.

4. heart.csv: The base dataset used for comparative data visualization and training reference.

# Installation
1. Clone the repository in your local machine: git clone https://github.com/sannidhya4114/AIML-Project-Vityarthi.git
2. Enter in your file AIML-Project-Vityarthi: cd AIML-Project-Vityarthi
3. Install Dependencies: Ensure you have Python 3.x installed, then run: pip install pandas joblib matplotlib seaborn

# How to Use
1. Run the application python Main.py
2. Input Data: Enter the clinical values when prompted (e.g., Age, Sex, Chest Pain Type, etc.).
3. View Results: The script will display:

   Diagnosis: "DISEASE DETECTED" or "NO DISEASE DETECTED."

   Risk Probability: A calculated percentage.

   Comparative Analysis: A popup window showing how the user's data compares to the average metrics in the dataset.
