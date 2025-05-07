This repository contains the implementation of an ensemble model for exploratory data analysis (EDA) and prediction of cardiomyopathy using machine learning techniques.

Project Overview

Cardiomyopathy is a serious heart muscle disease that can lead to heart failure. This project proposes an innovative approach to analyze cardiomyopathy datasets using EDA and machine learning classifiers (KNN, Decision Tree, Random Forest) to identify key disease-related features and improve predictive performance. The model aims to reduce false positives and negatives, aiding early diagnosis and risk assessment.

Prerequisites





Hardware:





Processor: Minimum Intel i3



RAM: Minimum 4 GB



Hard Disk: Minimum 250 GB



Software:





Python 3.8+



Google Colab or local IDE (e.g., VS Code, Anaconda)



Operating System: Windows, Linux
Methodology





Dataset Acquisition: Used the UCI Heart Disease dataset.



Exploratory Data Analysis (EDA):





Histogram plots for numeric features.



Count plot for target variable.



Correlation heatmap for feature relationships.



Data Preprocessing:





One-hot encoding for categorical variables (sex, cp, fbs, restecg, exang, slope, ca, thal).



Standardization of numeric features (age, trestbps, chol, thalach, oldpeak).



Train-test split (80% train, 20% test).



Modeling:





K-Nearest Neighbors (KNN) with k=5.



Decision Tree Classifier.



Random Forest Classifier.



Evaluation:





Metrics: Accuracy, precision, recall, F1-score.



Visualizations: Confusion matrices for each classifier.
