import numpy as np
import pandas as pd
datasetURL = "FamilyData3.csv"
dataset = pd.read_csv(datasetURL)
dataset.head()

# preprocessing on the data: converting from Categorical to Numerical
from sklearn.preprocessing import LabelEncoder
encoder = LabelEncoder()
dataset.Main_Source_of_Income = encoder.fit_transform(dataset.Main_Source_of_Income)
dataset.Household_Head_Sex = encoder.fit_transform(dataset.Household_Head_Sex)
dataset.Household_Head_Marital_Status = encoder.fit_transform(dataset.Household_Head_Marital_Status)
dataset.Household_Head_Job_or_Business_Indicator = encoder.fit_transform(dataset.Household_Head_Job_or_Business_Indicator)
dataset.Type_of_Household = encoder.fit_transform(dataset.Type_of_Household)

dataset.info()

#normalizing the data
def normalize(X):
    m, n = X.shape
    minValue =X.min()
    maxValue =X.max()
    X =(X-minValue)/(maxValue-minValue)
    return X;

#Data PreProcessing
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

# spliting data to input = x, output = y

x = dataset[["monthly_income", "Total_Income_from_Entrepreneurial_Acitivites","Agricultural_Household_indicator",
             "Total_Number_of_Family_members","Total_Number_of_Family Members Employed", "Number_of_Cellular_Phone"]]
y = dataset.iloc[:, -2]

#Split the data into train and test
xTrain, xTest, yTrain, yTest = train_test_split(normalize(x), y, test_size=0.25, random_state = 1224, shuffle = True)

print(xTrain.shape)
print(xTest.shape)
print(yTrain.shape)
print(yTest.shape)

x.describe()

from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.metrics import mean_squared_error

# Gradient Boosting Model
GBModel = GradientBoostingRegressor(n_estimators=100, max_depth=5, learning_rate=0.1, alpha=0.1, subsample=0.8,random_state=1224)

GBModel.fit(xTrain, yTrain)
yPred = GBModel.predict(xTest)
mse = mean_squared_error(yTest, yPred)
print("Mean Squared Error:", mse)
print('GB Regression Train Score is:', GBModel.score(xTrain, yTrain))
print('GB Regression Test Score is:', GBModel.score(xTest, yTest))
min = [517, 0, 0, 1, 0, 0]
max = [541566, 423247, 2, 26, 8, 10]

input = [22015, 2034, 0, 4, 1, 2]
for i in range(6):
    input[i] = (input[i] - min[i]) / (max[i] - min[i])
print(input)
input = [np.array(input)]
prediction = GBModel.predict(input)
print("Predicted output is: ", prediction)

# Perform cross-validation
cv_scores = cross_val_score(GBModel, x, y, cv=5, scoring='r2')

# Print the cross-validation scores
print("Cross-Validation Scores:", cv_scores)

import pickle
# Save the model
pickle.dump(GBModel, open('model_file.pkl', 'wb'))

