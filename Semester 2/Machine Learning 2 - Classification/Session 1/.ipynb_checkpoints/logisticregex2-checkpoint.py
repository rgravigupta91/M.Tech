import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Synthetic dataset of emails and corresponding labels (spam or not spam)
emails = [
    "Buy cheap watches now!",
    "Hello, how are you?",
    "Earn money fast!",
    "Free trial offer",
    "Get a new iPhone for free!",
    "Hey there!",
]

labels = [1, 0, 1, 1, 1, 0]  # 1 for spam, 0 for not spam

# Create a list of spam words (features)
spam_words = ["cheap", "money", "free"]

# Feature extraction function
def extract_features(emails, spam_words):
    features = []
    for email in emails:
        word_count = {word: email.lower().count(word) for word in spam_words}
        feature_vector = [word_count[word] for word in spam_words]
        features.append(feature_vector)
    return features

# Convert emails to feature vectors
X = extract_features(emails, spam_words)
y = labels

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42)

# Create a logistic regression model
model = LogisticRegression()

# Train the model using the training data
model.fit(X_train, y_train)

# Make predictions on the test data
y_pred = model.predict(X_test)

# Calculate the accuracy of the model
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")
