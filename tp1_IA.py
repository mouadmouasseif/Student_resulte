import pandas as pd
from sklearn.model_selection import train_test_split # type: ignore
from sklearn.linear_model import LogisticRegression

# Load the dataset
data = pd.read_csv('C:/Users/mouad/Desktop/python/student-mat.csv', sep=';')

# Select relevant features
features = ['studytime', 'failures', 'famsup', 'Medu', 'Fedu', 'traveltime', 'famrel', 'freetime', 'goout', 'Dalc', 'Walc', 'health', 'absences', 'G1', 'G2']
X = data[features]
y = data['G3']

# Convert categorical variables to numerical
X = pd.get_dummies(X, columns=['famsup'], drop_first=True)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

# Train the model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Show only the predictions
# Make predictions
y_pred = model.predict(X_test)

# Show predictions with student numbers
print("Predictions:")
for i, prediction in enumerate(y_pred):
    print(f"Student number: {i+1}, Grade prediction: {prediction}")
    