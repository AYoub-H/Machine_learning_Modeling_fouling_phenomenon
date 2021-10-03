import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import keras
from keras.layers import Dense, Activation
from keras.models import Sequential
from sklearn.model_selection import train_test_split
import sklearn.metrics, math
import matplotlib.pyplot as plt
from keras.callbacks import EarlyStopping
from keras.layers.normalization import BatchNormalization
from sklearn.linear_model import LinearRegression
import sklearn.metrics, math


# -----------------------------------------------------------------------------
# Define custom loss functions for regression in Keras 
# -----------------------------------------------------------------------------

# root mean squared error (rmse) for regression
def rmse(y_true, y_pred):
    from keras import backend
    return backend.sqrt(backend.mean(backend.square(y_pred - y_true), axis=-1))


# mean squared error (mse) for regression
def mse(y_true, y_pred):
    from keras import backend
    return backend.mean(backend.square(y_pred - y_true), axis=-1)


# coefficient of determination (R^2) for regression
def r_square(y_true, y_pred):
    from keras import backend as K
    SS_res = K.sum(K.square(y_true - y_pred))
    SS_tot = K.sum(K.square(y_true - K.mean(y_true)))
    return 1 - SS_res / (SS_tot + K.epsilon())


def r_square_loss(y_true, y_pred):
    from keras import backend as K
    SS_res = K.sum(K.square(y_true - y_pred))
    SS_tot = K.sum(K.square(y_true - K.mean(y_true)))
    return 1 - (1 - SS_res / (SS_tot + K.epsilon()))


# Importing the dataset
dataset = pd.read_csv("data/data_fouling_.csv", sep=";", decimal=",")
x = dataset.iloc[:, 0:7].values
y = dataset.iloc[:, 7].values

# Feature Scaling
sc = MinMaxScaler()
x = sc.fit_transform(x)
y = y.reshape(-1, 1)
y = sc.fit_transform(y)

# Splitting the dataset into the Training, Test, validation set
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.10, random_state=4)
x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=0.05, random_state=4)

# built Keras sequential model 
model = Sequential()
# add batch normalization
model.add(BatchNormalization())
# Adding the input layer:
model.add(Dense(7, input_dim=x_train.shape[1], activation='relu'))
# the first hidden layer:
model.add(Dense(12, activation='relu'))
# Adding the output layer
model.add(Dense(1, activation='sigmoid'))

# Compiling the ANN
model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mse', 'mae', rmse, r_square])

# enable early stopping based on mean_squared_error
earlystopping = EarlyStopping(monitor="mse", patience=40, verbose=1, mode='auto')

# Fit model
result = model.fit(x_train, y_train, batch_size=100, epochs=1000, validation_data=(x_test, y_test),
                   callbacks=[earlystopping])

# get predictions
y_pred = model.predict(x_test)

# -----------------------------------------------------------------------------
# print statistical figures of merit
# -----------------------------------------------------------------------------
print("Mean absolute error (MAE):      %f" % sklearn.metrics.mean_absolute_error(y_test, y_pred))
print("Mean squared error (MSE):       %f" % sklearn.metrics.mean_squared_error(y_test, y_pred))
print("Root mean squared error (RMSE): %f" % math.sqrt(sklearn.metrics.mean_squared_error(y_test, y_pred)))
print("R square (R^2):                 %f" % sklearn.metrics.r2_score(y_test, y_pred))

# -----------------------------------------------------------------------------
# Plot learning curves including R^2 and RMSE
# -----------------------------------------------------------------------------

# plot training curve for R^2 (beware of scale, starts very low negative)
plt.plot(result.history['val_r_square'])
plt.plot(result.history['r_square'])
plt.title('model R^2')
plt.ylabel('R^2')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

# plot training curve for rmse
plt.plot(result.history['rmse'])
plt.plot(result.history['val_rmse'])
plt.title('rmse')
plt.ylabel('rmse')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

# print the linear regression and display datapoints 
regressor = LinearRegression()
regressor.fit(y_test.reshape(-1, 1), y_pred)
y_fit = regressor.predict(y_pred)

reg_intercept = round(regressor.intercept_[0], 4)
reg_coef = round(regressor.coef_.flatten()[0], 4)
reg_label = "y = " + str(reg_intercept) + "*x +" + str(reg_coef)

plt.scatter(y_test, y_pred, color='blue', label='data')
plt.plot(y_pred, y_fit, color='red', linewidth=2, label='Linear regression\n' + reg_label)
plt.title('Linear Regression')
plt.legend()
plt.xlabel('observed')
plt.ylabel('predicted')
plt.show()
