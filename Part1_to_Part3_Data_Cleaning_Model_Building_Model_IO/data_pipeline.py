import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import accuracy_score

"""# Data Collection, Cleaning and Analysis

### Diabetes Dataset

The objective of the dataset is to diagnostically predict whether or not a patient has diabetes, based on certain diagnostic measurements included in the dataset. Several constraints were placed on the selection of these instances from a larger database. In particular, all patients here are females at least 21 years old of Pima Indian heritage.

The datasets consists of several medical predictor variables and one target variable, Outcome. Predictor variables includes:
* Pregnancies: Number of pregnancies, which can
impact diabetes risk.
* Glucose: Blood sugar level after a glucose tolerance test.
* BloodPressure: Diastolic blood pressure, linked to cardiovascular health.
* SkinThickness: Thickness of triceps skinfold, used to estimate body fat.
* Insulin: 2-hour serum insulin level, indicating glucose regulation.
* BMI: Body mass index, a measure of body fat based on height and weight.
* DiabetesPedigreeFunction: Genetic likelihood of diabetes based on family history.
* Age: Age of the individual, with older age increasing diabetes risk.
"""

#loading the diabetes dataset to a pandas DataFrame
diabetes_dataset = pd.read_csv('/content/drive/MyDrive/diabetes.csv')

# printing the first 5 rows of the dataset
diabetes_dataset.head()

# Number of Rows and Columns in this dataset
diabetes_dataset.shape

diabetes_dataset.info()

diabetes_dataset.duplicated().sum()

# getting the statistical measures of the data
diabetes_dataset.describe()

diabetes_dataset['Outcome'].value_counts()

"""0 --> Non-Diabetic

1 --> Diabetic
"""

diabetes_dataset.groupby('Outcome').mean()

correlation_matrix = diabetes_dataset.corr()
correlation_matrix

# separating the data and labels
X = diabetes_dataset.drop(columns = 'Outcome', axis=1)
Y = diabetes_dataset['Outcome']

X

Y

scaler = StandardScaler()

scaler.fit(X)

standardized_data = scaler.transform(X)
standardized_data

X = standardized_data
Y = diabetes_dataset['Outcome']

X

"""# Train - Test Split"""

X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size = 0.2, stratify=Y, random_state=2)

print(X.shape, X_train.shape, X_test.shape)

"""# Training the Model"""

classifier = svm.SVC(kernel='linear')

#training the support vector Machine Classifier
classifier.fit(X_train, Y_train)

"""# Model Evaluation

## Accuracy Score
"""

# accuracy score on the training data
X_train_prediction = classifier.predict(X_train)
training_data_accuracy = accuracy_score(X_train_prediction, Y_train)

print('Accuracy score of the training data : ', training_data_accuracy)

# accuracy score on the test data
X_test_prediction = classifier.predict(X_test)
test_data_accuracy = accuracy_score(X_test_prediction, Y_test)

print('Accuracy score of the test data : ', test_data_accuracy)

"""#Making a Predictive System"""

input_data = (5,166,72,19,175,25.8,0.587,51)

# changing the input_data to numpy array
input_data_as_numpy_array = np.asarray(input_data)

# reshape the array as we are predicting for one instance
input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)

# standardize the input data
std_data = scaler.transform(input_data_reshaped)
print(std_data)

prediction = classifier.predict(std_data)
print(prediction)

if (prediction[0] == 0):
  print('The person is not diabetic')
else:
  print('The person is diabetic')

"""# Saving the trained model"""

import joblib

filename = 'trained_model.sav'
joblib.dump(classifier, '/content/drive/MyDrive/Diabetes Model/filename')

# Save the scaler to a file
joblib.dump(scaler, 'scaler.joblib')

"""## Loading the saved model"""

loaded_model = joblib.load('/content/drive/MyDrive/Diabetes Model/filename')

input_data = (
    int(input("Enter number of pregnancies: ")),
    float(input("Enter glucose level: ")),
    float(input("Enter blood pressure: ")),
    float(input("Enter skin thickness: ")),
    float(input("Enter insulin level: ")),
    float(input("Enter BMI: ")),
    float(input("Enter Diabetes Pedigree Function: ")),
    int(input("Enter age: "))
)
# changing the input_data to numpy array
input_data_as_numpy_array = np.asarray(input_data)

# reshape the array as we are predicting for one instance
input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)

# standardize the input data
std_data = scaler.transform(input_data_reshaped)
print(std_data)

prediction = loaded_model.predict(std_data)
print(prediction)

if (prediction[0] == 0):
  print('The person is not diabetic')
else:
  print('The person is diabetic')
