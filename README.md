# SVM-salary-data-problem
Prepare a classification model using SVM for salary data 
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from pandas.plotting import scatter_matrix

# Load the train and test data
train_data = pd.read_csv('SalaryData_Train(1).csv')
test_data = pd.read_csv('SalaryData_Test(1).csv')

# Separate the features (X) and the target variable (y) in both the train and test datasets
X_train = train_data.drop('Salary', axis=1)
y_train = train_data['Salary']
X_test = test_data.drop('Salary', axis=1)
y_test = test_data['Salary']

# Perform data preprocessing

# Encode categorical features to numerical labels
label_encoder = LabelEncoder()
categorical_columns = ['workclass', 'education', 'maritalstatus', 'occupation', 'relationship', 'race', 'sex', 'native']
for column in categorical_columns:
    X_train[column] = label_encoder.fit_transform(X_train[column])
    X_test[column] = label_encoder.transform(X_test[column])

# Scale the numerical features using standardization
scaler = StandardScaler()
numerical_columns = ['age', 'educationno', 'capitalgain', 'capitalloss', 'hoursperweek']
X_train[numerical_columns] = scaler.fit_transform(X_train[numerical_columns])
X_test[numerical_columns] = scaler.transform(X_test[numerical_columns])

# Visualize univariate analysis (histograms)
def visualize_univariate(data):
    for col in data.columns:
        plt.figure(figsize=(8, 5))
        sns.histplot(data[col], kde=True)
        plt.title(f'Distribution of {col}')
        plt.xlabel(col)
        plt.ylabel('Frequency')
        plt.show()

# Visualize bivariate analysis (scatter plots)
def visualize_bivariate(data):
    scatter_matrix(data, figsize=(15, 15))
    plt.suptitle('Pairwise Scatter Plot of Features', size=20)
    plt.show()

# Create SVM models with different kernels
kernels = ['linear', 'poly', 'rbf', 'sigmoid']
for kernel in kernels:
    print(f"\nTraining SVM with {kernel} kernel:")
    # Create an SVM model and train it
    svm = SVC(kernel=kernel)
    svm.fit(X_train, y_train)

    # Make predictions on the test set
    y_pred = svm.predict(X_test)

    # Calculate accuracy and print classification report
    accuracy = accuracy_score(y_test, y_pred)
    classification_rep = classification_report(y_test, y_pred)
    print('Accuracy:', accuracy)
    print('Classification Report:\n', classification_rep)

# Visualize univariate analysis
visualize_univariate(X_train)

# Visualize bivariate analysis
visualize_bivariate(X_train)
