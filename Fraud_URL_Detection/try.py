import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Load the dataset
data = pd.read_csv("phishing.csv")
data = data.drop(['Index'],axis = 1)
# Create feature and target vectors
X = data.drop(["class"], axis=1)
y = data["class"]

# Convert URLs to feature vectors



# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the SVM model using the training set
svm_model = SVC(kernel='linear')
svm_model.fit(X_train, y_train)

# Train the Decision Tree model using the training set
dt_model = DecisionTreeClassifier(random_state=42)
dt_model.fit(X_train, y_train)

# Make predictions using both models
svm_preds = svm_model.predict(X_test)
dt_preds = dt_model.predict(X_test)

# Combine the predictions using a simple voting mechanism
ensemble_preds = []
for i in range(len(X_test)):
    if svm_preds[i] == dt_preds[i]:
        ensemble_preds.append(svm_preds[i])
    else:
        ensemble_preds.append(np.random.choice([svm_preds[i], dt_preds[i]]))

# Compute evaluation metrics for the ensemble model
accuracy = accuracy_score(y_test, ensemble_preds)
precision = precision_score(y_test, ensemble_preds, average='weighted')
recall = recall_score(y_test, ensemble_preds, average='weighted')
f1 = f1_score(y_test, ensemble_preds, average='weighted')

# Print the evaluation metrics
print("Accuracy: {:.2f}%".format(accuracy*100))
print("Precision: {:.2f}%".format(precision*100))
print("Recall: {:.2f}%".format(recall*100))
print("F1 Score: {:.2f}%".format(f1*100))
