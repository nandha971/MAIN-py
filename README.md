# MAIN-py
# Loan Approval Prediction using Decision Tree in Colab

# Step 1: Import Libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier, export_text, plot_tree
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt

# Step 2: Create a Sample Dataset (50 records)
data = {
    'Gender': ['Male','Female','Male','Male','Female','Female','Male','Male','Female','Male',
               'Male','Female','Male','Female','Male','Male','Female','Male','Female','Male',
               'Female','Male','Female','Male','Male','Female','Male','Female','Male','Male',
               'Male','Female','Male','Male','Female','Male','Female','Male','Male','Female',
               'Female','Male','Male','Female','Male','Male','Female','Male','Female','Male'],
    
    'Married': ['Yes','No','Yes','No','Yes','No','Yes','No','Yes','Yes',
                'No','Yes','No','Yes','Yes','No','Yes','No','Yes','Yes',
                'No','Yes','No','Yes','Yes','No','Yes','No','Yes','Yes',
                'No','Yes','No','Yes','Yes','No','Yes','No','Yes','Yes',
                'No','Yes','No','Yes','Yes','No','Yes','No','Yes','Yes'],
    
    'ApplicantIncome': [5000,3000,4000,6000,2500,3500,8000,2000,4500,7000,
                        5200,2800,3900,4200,6500,3100,5700,4800,2600,7500,
                        3300,4100,2900,8000,3600,2500,5500,6000,4500,3000,
                        4000,6800,2700,7200,5000,3100,4700,3400,8100,2600,
                        2900,4300,3700,6900,5200,2700,3500,7800,2500,4600],
    
    'LoanAmount': [200,100,150,250,120,110,300,90,180,220,
                   210,95,140,160,260,130,200,170,115,280,
                   125,190,105,310,135,100,230,240,180,120,
                   160,270,110,290,200,140,180,150,320,115,
                   95,175,130,275,205,120,150,300,105,185],
    
    'Credit_History': [1,0,1,1,0,1,1,0,1,1,
                       0,1,1,1,1,0,1,1,0,1,
                       0,1,0,1,1,0,1,1,1,0,
                       1,1,0,1,1,0,1,0,1,0,
                       0,1,1,1,1,0,1,1,0,1],
    
    'Loan_Status': ['Y','N','Y','Y','N','N','Y','N','Y','Y',
                    'N','Y','Y','Y','Y','N','Y','Y','N','Y',
                    'N','Y','N','Y','Y','N','Y','Y','Y','N',
                    'Y','Y','N','Y','Y','N','Y','N','Y','N',
                    'N','Y','Y','Y','Y','N','N','Y','N','Y']
}

# Step 3: Convert to DataFrame
df = pd.DataFrame(data)
print("Dataset Shape:", df.shape)
print("\nFirst 5 rows:\n", df.head())

# Step 4: Encode Categorical Variables
le = LabelEncoder()
for col in ['Gender', 'Married', 'Loan_Status']:
    df[col] = le.fit_transform(df[col])

# Step 5: Split Data into Features & Target
X = df.drop('Loan_Status', axis=1)
y = df['Loan_Status']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# Step 6: Train Decision Tree Model
clf = DecisionTreeClassifier(criterion="entropy", max_depth=4, random_state=42)
clf.fit(X_train, y_train)

# Step 7: Predictions
y_pred = clf.predict(X_test)

# Step 8: Evaluation
print("\nAccuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Step 9: Show Decision Tree Rules
rules = export_text(clf, feature_names=list(X.columns))
print("\nDecision Tree Rules:\n", rules)

# Step 10: Visualize the Decision Tree
plt.figure(figsize=(12,6))
plot_tree(clf, feature_names=X.columns, class_names=["Rejected","Approved"], filled=True, rounded=True)
plt.show()

# Step 11: Example Prediction
sample = [[1, 1, 4000, 150, 1]]  # Male, Married, Income=4000, Loan=150, Good Credit
pred = clf.predict(sample)
print("\nLoan Approval Prediction for sample:", "Approved" if pred[0] == 1 else "Rejected")

Dataset Shape: (50, 6)

First 5 rows:
    Gender Married  ApplicantIncome  LoanAmount  Credit_History Loan_Status
0    Male     Yes             5000         200               1           Y
1  Female      No             3000         100               0           N
2    Male     Yes             4000         150               1           Y
3    Male      No             6000         250               1           Y
4  Female     Yes             2500         120               0           N

Accuracy: 1.0

Classification Report:
               precision    recall  f1-score   support

           0       1.00      1.00      1.00         7
           1       1.00      1.00      1.00         8

    accuracy                           1.00        15
   macro avg       1.00      1.00      1.00        15
weighted avg       1.00      1.00      1.00        15


Decision Tree Rules:
 |--- Credit_History <= 0.50
|   |--- class: 0
|--- Credit_History >  0.50
|   |--- ApplicantIncome <= 3550.00
|   |   |--- ApplicantIncome <= 3150.00
|   |   |   |--- class: 1
|   |   |--- ApplicantIncome >  3150.00
|   |   |   |--- class: 0
|   |--- ApplicantIncome >  3550.00
|   |   |--- class: 1



Loan Approval Prediction for sample: Approved
/usr/local/lib/python3.12/dist-packages/sklearn/utils/validation.py:2739: UserWarning: X does not have valid feature names, but DecisionTreeClassifier was fitted with feature names
  warnings.warn(
