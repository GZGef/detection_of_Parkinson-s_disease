# Parkinson's Disease Prediction using XGBoost

This project is focused on early-stage prediction of Parkinson's Disease using machine learning. We use the XGBoost algorithm for classification and `sklearn` for data preprocessing and normalization.

## What is Parkinson’s Disease?

Parkinson’s Disease is a progressive disorder of the central nervous system that affects movement and often causes tremors and stiffness. It has 5 stages and affects over 1 million people annually in India alone. It is a chronic, currently incurable neurodegenerative disorder that primarily targets dopamine-producing neurons in the brain.

## Goal

The goal of this project is to build a machine learning model that can predict whether a person has Parkinson’s Disease based on biomedical voice measurements.

## Dataset

We use the **Parkinson’s dataset** from the [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/parkinsons). It contains various features extracted from voice recordings. Each row represents a voice sample, and the task is to classify whether the individual has Parkinson’s disease.

## Features

The dataset includes a number of biomedical voice measurements such as:
- MDVP:Fo(Hz), MDVP:Fhi(Hz), MDVP:Flo(Hz)
- MDVP:Jitter(%), MDVP:RAP, MDVP:PPQ
- Jitter:DDP, MDVP:Shimmer, Shimmer:APQ3, etc.

The target column is `status`:
- `1` – Parkinson’s Disease
- `0` – Healthy

## Steps to Run the Project

1. **Install dependencies**:
```bash
pip install xgboost scikit-learn pandas numpy matplotlib seaborn
```

2. **Load and preprocess data**:
- Load the UCI Parkinson’s dataset
- Normalize the features using `StandardScaler` from `sklearn.preprocessing`
- Split the dataset into training (80%) and testing (20%) sets

3. **Train the model**:
- Use `XGBClassifier` from the `xgboost` library to train the model

4. **Evaluate the model**:
- Calculate accuracy on the test set
- Try to achieve accuracy greater than **95%** for bonus points

5. **Visualize results**:
- Display confusion matrix, ROC curve, and other relevant metrics

## What is XGBoost?

**XGBoost (Extreme Gradient Boosting)** is a high-performance implementation of gradient boosted decision trees. It is renowned for its speed and performance, often being the top choice in machine learning competitions like Kaggle.

There are two popular ways to use XGBoost:
- **Learning API** – A powerful low-level interface with built-in cross-validation
- **Scikit-Learn API** – A wrapper interface compatible with `scikit-learn` pipelines and syntax

In this project, we use the `XGBClassifier` from `xgboost`, which follows the Scikit-Learn API for easy integration.

## Recommendation

> 💡 **Run the `main.py` script in [Google Colab](https://colab.research.google.com/) for better visualization and an interactive experience.**
