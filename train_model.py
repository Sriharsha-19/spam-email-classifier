import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score
import pickle


# Load the dataset
data = pd.read_csv("spam.csv")


# Get messages and their labels
X = data["message"]
y = data["label"]


# Split the data into training and testing data
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)


# Create the machine learning pipeline
model = Pipeline([
    ("tfidf", TfidfVectorizer()),
    ("classifier", MultinomialNB())
])


# Train the model
model.fit(X_train, y_train)


# Test the model
predictions = model.predict(X_test)


# Calculate accuracy
accuracy = accuracy_score(y_test, predictions)

print("Model Accuracy:", accuracy)


# Save the trained model
with open("spam_model.pkl", "wb") as file:
    pickle.dump(model, file)


print("Model saved successfully!")