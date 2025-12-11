# Import dependencies
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pandas as pd

# Get data from local CSV
df = pd.read_csv("data/Iris.csv", index_col=0)

# Transform response (Species) for XGBClassifier
def transform(val):
    if val == "Iris-setosa":
        return 0
    elif val == "Iris-versicolor":
        return 1
    else:
        return 2

df["Species"] = df["Species"].apply(transform)

X = df[["SepalLengthCm", "SepalWidthCm", "PetalLengthCm", "PetalWidthCm"]]
y = df["Species"]

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train the XGBoost classifier
model = xgb.XGBClassifier(objective='multi:softmax', eval_metric='merror')
model.fit(X_train, y_train)

# Make predictions on test set
predictions = model.predict(X_test)

# Evaluate the model
train_accuracy = accuracy_score(y_test, predictions)
print(f"Accuracy: {train_accuracy}")