import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score

# Load the dataset
data = pd.read_csv('creditcard.csv')
# print(data.columns)
# print(data.head(5))

# Visualize the distribution of fraud vs. non-fraud (optional)
#sns.countplot(x="Class", data=data)
#plt.show()cls

#Check for class distribution
print(data['Class'].value_counts())

# Feature preprocessing
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
data['normalized_amount'] = scaler.fit_transform(data['Amount'].values.reshape(-1, 1))
data.drop(['Amount'], axis=1, inplace=True)

# Split data into features (X) and target (y)
X = data.drop(['Class'], axis=1)
y = data['Class']

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train the model
model = RandomForestClassifier(n_estimators=100, random_state=42)
print("Training started")
model.fit(X_train, y_train)
print("Training completed")

# Make predictions and evaluate the model
y_pred = model.predict(X_test)
print("Classification Report:\n", classification_report(y_test, y_pred))

import joblib

# Save the trained model
joblib.dump(model, "app/model.pkl")


