# GreenGrid AI

## AI-Based Carbon Emission Reduction Prediction

GreenGrid AI is an AI/ML-based building energy analytics system designed to analyze energy, environmental, and building-related parameters and predict carbon-emission reduction levels.

The system classifies buildings into four categories:

- No Reduction
- Low Reduction
- Moderate Reduction
- High Reduction

## Project Objectives

- Analyze building energy consumption and environmental parameters.
- Identify patterns associated with carbon emissions.
- Predict carbon-emission reduction categories.
- Compare multiple machine-learning classification models.
- Develop a proposed Neuro-Tree Fusion approach.

## Machine Learning Models

The project implements and compares:

- Logistic Regression
- XGBoost
- Extra Trees Classifier
- Proposed Neuro-Tree Fusion Model

## Methodology

The project includes:

1. Data preprocessing
2. Exploratory Data Analysis
3. Feature processing
4. Model training
5. Model evaluation
6. Carbon-emission reduction prediction

## Model Evaluation

Models were evaluated using:

- Accuracy
- Precision
- Recall
- F1-Score
- Confusion Matrix
- ROC-AUC

## Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- XGBoost
- TensorFlow/Keras
- Matplotlib
- Seaborn
- Tkinter
- Redis

## Project Architecture

![System Architecture](architectures/system.jpg)

## Exploratory Data Analysis

![EDA Results](results/eda_plots.png)

## Proposed Model

![Proposed Architecture](architectures/Proposed.png)

## Results

The repository contains confusion matrices and ROC curves for the implemented machine-learning models.

## Project Structure

```text
GreenGrid-AI/
├── src/
├── architectures/
├── results/
├── README.md
├── requirements.txt
└── .gitignore
