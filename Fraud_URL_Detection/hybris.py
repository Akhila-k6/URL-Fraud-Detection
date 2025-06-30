import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

# Load the dataset
url_data = pd.read_csv('phishing.csv')

# Split the dataset into input features and target variable
X = url_data.drop(['class'],axis =1)
y = url_data['class']

# Convert the input features into numerical values using CountVectorizer
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(X)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X.toarray(), y, test_size=0.2, random_state=42)

# Train the SVM model using the training set
svm = SVC(kernel='linear', probability=True)
svm.fit(X_train, y_train)

# Make predictions using the SVM model
svm_predictions = svm.predict(X_test)

# Train the decision tree model using the training set
tree = DecisionTreeClassifier(random_state=42)
tree.fit(X_train, y_train)

# Make predictions using the decision tree model
tree_predictions = tree.predict(X_test)

# Combine the predictions using a weighted average
weighted_predictions = (svm.predict_proba(X_test)[:,1] + tree.predict_proba(X_test)[:,1]) / 2
final_predictions = np.round(weighted_predictions)

# Calculate the accuracy of the hybrid model
accuracy = accuracy_score(y_test, final_predictions)
print("Accuracy:", accuracy)
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Predict the classes of the test set
y_pred = clf.predict(X_test)

# Compute the metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

# Print the results
print("Accuracy: {:.2f}%".format(accuracy * 100))
print("Precision: {:.2f}%".format(precision * 100))
print("Recall: {:.2f}%".format(recall * 100))
print("F1 score: {:.2f}%".format(f1 * 100))
