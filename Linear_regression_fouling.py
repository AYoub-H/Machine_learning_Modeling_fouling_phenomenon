# Importing libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as seabornInstance
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics

# Importing dataset
dataset = pd.read_csv("data/data_fouling.csv", sep=';', decimal=',')
dataset.head()

# data mining
dataset.shape

# statistical details of the dataset
dataset.describe()

# Dividing data
X = dataset[['Fdw', 'Tdw', 'Ta', 'Thwi', 'Pa', 'RH']]
y = dataset['Thwo']

# Check the mean value of the column 'Thwo'.
plt.figure(figsize=(15, 10))
plt.tight_layout()
seabornInstance.distplot(dataset['Thwo'])

# 80% training set, 20 % testing set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Model
regressor = LinearRegression()
regressor.fit(X_train, y_train)

# In case of multivariable linear regression, the regression model must find the most optimal coefficients for all
# attributes. To see which coefficients our regression model to choose:
coeff_df = pd.DataFrame(regressor.coef_, X.columns, columns=['Coefficient'])
coeff_df

# Intercept calculation
regressor.intercept_

# Predictions of test data
y_pred = regressor.predict(X_test)
y_pred

df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
df1 = df.head(25)
df1

# Comparison of actual and expected values
df1.plot(kind='bar', figsize=(10, 8))
plt.grid(which='major', linestyle='-', linewidth='0.5', color='green')
plt.grid(which='minor', linestyle=':', linewidth='0.5', color='red')
plt.show()

print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
print('R-Squared:', regressor.score(X_train, y_train))
