#import essential libraries
import numpy as np
import pandas as pd
import joblib

#lib for visualizations
import matplotlib.pyplot as plt
import seaborn as sns

#import the dataset
from sklearn.datasets import load_breast_cancer

#load the dataset
cancer = load_breast_cancer()

#convert into dataframe using pandas library
data = pd.DataFrame(cancer.data, columns= cancer.feature_names)
#add target col
data['target'] = cancer.target

#view basic information
#Exploratory data Analysis
print(data.head())
print("Shape of Data: ", data.shape)
print(data.info())
print(data.describe())
print("\nMissing value to each column: \n",data.isnull().sum())

#Dealing with missing values
#data = data.fillna(data.mean()) this is for replacing with mean
#data = data.fillna(data.median()) replace missing value with median
#data = data.dropna() this drops the rows with missing values
#for the current data we are working on we have no missing value so we ignore


#correlation of the target with features of input
import matplotlib.pyplot as plt
import seaborn as sns

#compute correlation matrix
#this calculates the relation between all columns
corr_matrix = data.corr(numeric_only=True)

# Visualize with heatmap
plt.figure(figsize=(14, 12))
sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="coolwarm", square=False)
plt.title("Correlation Heatmap")
plt.show()

# Calculate correlation with target
correlation_with_target = data.corr()['target'].drop('target')

# Sort by absolute correlation (highest to lowest)
top_features = correlation_with_target.abs().sort_values(ascending=False)
print(top_features)

# Select top 10 features
selected_features = top_features.head(9).index.tolist()
print("Selected Features:", selected_features)
joblib.dump(selected_features, "models/selected_features.pkl")


X = data[selected_features]
y = data['target']

#we then scale the data
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)  

#split the data into training and testing
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=45, stratify=y
)
#stratify = y ensures that count of y value remain balanced in the spliited groups

#train logistic regression
from sklearn.linear_model import LogisticRegression

logreg_model = LogisticRegression()
logreg_model.fit(X_train, y_train)


#train MLP
from sklearn.neural_network import MLPClassifier

mlp_model = MLPClassifier(hidden_layer_sizes=(64, 32), activation='relu', max_iter=1000, random_state=45)
#we have two hidden layer with 64 nodes then 32 nodes 
#activation relu means the activation function used on each nodes is rectified linear unit f(x) = max(0,x)
#max_iter is the number of iterations it goes through the data and updates the weight upto 1000 times
mlp_model.fit(X_train, y_train)


#make prediction
y_pred_logreg = logreg_model.predict(X_test)
y_pred_mlp = mlp_model.predict(X_test)

#evaluate the two model
#import evaluation metrics
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

#we then avaluate the logistic regression model and print
print("\nLogistic Regression Evaluation:")
print("Accuracy: ", accuracy_score(y_test, y_pred_logreg))
print("Precision:", precision_score(y_test, y_pred_logreg))
print("Recall:   ", recall_score(y_test, y_pred_logreg))
print("F1 Score: ", f1_score(y_test, y_pred_logreg))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_logreg))

#we evaluate and print the MLP classifier model
print("\nMLP Classifier Evaluation:")
print("Accuracy: ", accuracy_score(y_test, y_pred_mlp))
print("Precision:", precision_score(y_test, y_pred_mlp))
print("Recall:   ", recall_score(y_test, y_pred_mlp))
print("F1 Score: ", f1_score(y_test, y_pred_mlp))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_mlp))


#we then plot the comparision between the two models
plt.figure(figsize=(12, 5))

# Get confusion matrices
cm_logreg = confusion_matrix(y_test, y_pred_logreg)
cm_mlp = confusion_matrix(y_test, y_pred_mlp)

# Logistic Regression
plt.subplot(1, 2, 1)
sns.heatmap(cm_logreg, annot=True, fmt='d', cmap='Blues', cbar=False)
plt.title('Logistic Regression')
plt.xlabel('Predicted')
plt.ylabel('Actual')

# MLP Classifier
plt.subplot(1, 2, 2)
sns.heatmap(cm_mlp, annot=True, fmt='d', cmap='Greens', cbar=False)
plt.title('MLP Classifier')
plt.xlabel('Predicted')
plt.ylabel('Actual')

plt.tight_layout()
plt.show()



# Save Logistic Regression model
joblib.dump(logreg_model, "models/logistic_regression_model.pkl")

# Save MLP Classifier model
joblib.dump(mlp_model, "models/mlp_classifier_model.pkl")

# Save the scaler
joblib.dump(scaler, "models/scaler.pkl")
