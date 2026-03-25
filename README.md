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
1. Clone the repository:
