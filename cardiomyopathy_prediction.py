import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Load the dataset
df = pd.read_csv('heart.csv')

# Basic data exploration
print("Dataset Shape:", df.shape)
print("Columns:", df.columns)
print("Data Types:\n", df.dtypes)
print("First 5 Rows:\n", df.head())
print("Last 5 Rows:\n", df.tail())
print("Null Values:\n", df.isnull().any())
print("Dataset Info:\n", df.info())
print("Basic Statistics:\n", df.describe())

# EDA: Histogram for all numeric features
plt.figure(figsize=(15, 15))
df.hist()
plt.tight_layout()
plt.savefig('figures/histogram_all_features.png')
plt.close()

# EDA: Count plot for target variable
sns.countplot(x='target', data=df)
plt.xlabel('Target')
plt.ylabel('Count')
plt.title('Target Variable Distribution')
plt.savefig('figures/target_distribution.png')
plt.close()

# EDA: Correlation heatmap
corr_matrix = df.corr()
plt.figure(figsize=(20, 20))
sns.heatmap(corr_matrix, annot=True, cmap='RdYlGn')
plt.title('Correlation Heatmap')
plt.savefig('figures/correlation_heatmap.png')
plt.close()

# Data Preprocessing: One-hot encoding for categorical variables
dataset = pd.get_dummies(df, columns=['sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'ca', 'thal'])
print("Columns after encoding:", dataset.columns)

# Data Preprocessing: Standardize numeric features
scaler = StandardScaler()
columns_to_scale = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']
dataset[columns_to_scale] = scaler.fit_transform(dataset[columns_to_scale])
print("First 5 Rows after preprocessing:\n", dataset.head())

# Split dataset into features and target
X = dataset.drop('target', axis=1)
y = dataset['target']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

# KNN Classifier
knn = KNeighborsClassifier(n_neighbors=5, weights='uniform', algorithm='auto', leaf_size=30, p=2, metric='minkowski')
knn.fit(X_train, y_train)
y_pred_knn = knn.predict(X_test)
print("KNN Accuracy:", accuracy_score(y_test, y_pred_knn) * 100)
print("KNN Classification Report:\n", classification_report(y_test, y_pred_knn))
cm_knn = confusion_matrix(y_test, y_pred_knn)
plt.figure(figsize=(8, 6))
sns.heatmap(cm_knn, annot=True, fmt="d", cmap="Blues")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("KNN Confusion Matrix")
plt.savefig('figures/knn_confusion_matrix.png')
plt.close()

# Decision Tree Classifier
dt = DecisionTreeClassifier(random_state=42)
dt.fit(X_train, y_train)
y_pred_dt = dt.predict(X_test)
print("Decision Tree Accuracy:", accuracy_score(y_test, y_pred_dt) * 100)
print("Decision Tree Classification Report:\n", classification_report(y_test, y_pred_dt))
cm_dt = confusion_matrix(y_test, y_pred_dt)
plt.figure(figsize=(8, 6))
sns.heatmap(cm_dt, annot=True, fmt="d", cmap="Blues")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Decision Tree Confusion Matrix")
plt.savefig('figures/dt_confusion_matrix.png')
plt.close()

# Random Forest Classifier
rf = RandomForestClassifier(random_state=42)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)
print("Random Forest Accuracy:", accuracy_score(y_test, y_pred_rf) * 100)
print("Random Forest Classification Report:\n", classification_report(y_test, y_pred_rf))
cm_rf = confusion_matrix(y_test, y_pred_rf)
plt.figure(figsize=(8, 6))
sns.heatmap(cm_rf, annot=True, fmt="d", cmap="Blues")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Random Forest Confusion Matrix")
plt.savefig('figures/rf_confusion_matrix.png')
plt.close()
