# Import dependencies
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns

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

# Make predictions for the full dataset
all_predictions = model.predict(X)

# Build processed dataset: features, encoded label, prediction, and split
df_processed = X.copy()
df_processed["SpeciesEncoded"] = y
df_processed["PredictedClass"] = all_predictions
df_processed["Split"] = "train"
df_processed.loc[X_test.index, "Split"] = "test"

# Save single processed CSV
df_processed.to_csv("data/Iris_processed.csv", index=False)

# Make predictions on test set
predictions = model.predict(X_test)

feature_importance = model.feature_importances_
features = X.columns

plt.figure(figsize=(8, 5))
plt.barh(features, feature_importance)
plt.xlabel('Importance')
plt.title('Feature Importance in XGBoost Model')
plt.tight_layout()
plt.show()

# Confusion matrix to see prediction performance by class

cm = confusion_matrix(y_test, predictions)

plt.figure(figsize=(7, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['Setosa', 'Versicolor', 'Virginica'],
            yticklabels=['Setosa', 'Versicolor', 'Virginica'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.tight_layout()
plt.show()