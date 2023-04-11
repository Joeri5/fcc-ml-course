import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from imblearn.over_sampling import RandomOverSampler
from sklearn.preprocessing import StandardScaler
import copy
import seaborn as sns
import tensorflow as tf
from sklearn.linear_model import LinearRegression
import os
from utils.helper_functions import dir_except_checkpoints

# Dataset:
# Dua, D. and Graff, C. (2019). UCI Machine Learning
# Repository http://archive.ics.uci.edu/ml. Irvine, CA:
# University of California, School of Information and Computer Science.

# Source: Data Source: http://data.seoul.go.kr/
# SOUTH KOREA PUBLIC HOLIDAYS. URL: http://publicholidays.go.kr

dataset_cols = ["bike_count", "hour", "temp", "humidity", "wind", "visibility", "dew_pt_temp", "radiation", "rain", "snow", "functional"]
df = pd.read_csv("data/SeoulBikeData.csv").drop(["Date", "Holiday", "Seasons"], axis=1)

df.columns = dataset_cols
df["functional"] = (df["functional"] == "Yes").astype(int)
df = df[df["hour"] == 12]
df = df.drop(["hour"], axis=1)

print(df.head())

dir_except_checkpoints('bike_count_plot')

for label in df.columns[1:]:
    plt.scatter(df[label], df["bike_count"])
    plt.title(label)
    plt.ylabel("Bike Count at Noon")
    plt.xlabel(label)
    plt.savefig(os.path.join('bike_count_plot', f'plot_{label}.png'))
    plt.clf()
    
df = df.drop(["wind", "visibility", "functional"], axis=1)

# Train, Valid, Test Data
train, val, test = np.split(df.sample(frac=1), [int(0.6*len(df)), int(0.8*len(df))])

def get_xy(dataframe, y_label, x_labels=None):
  dataframe = copy.deepcopy(dataframe)
  if x_labels is None:
    X = dataframe[[c for c in dataframe.columns if c!=y_label]].values
  else:
    if len(x_labels) == 1:
      X = dataframe[x_labels[0]].values.reshape(-1, 1)
    else:
      X = dataframe[x_labels].values

  y = dataframe[y_label].values.reshape(-1, 1)
  data = np.hstack((X, y))

  return data, X, y

_, X_train_temp, y_train_temp = get_xy(train, "bike_count", x_labels=["temp"])
_, X_val_temp, y_val_temp = get_xy(val, "bike_count", x_labels=["temp"])
_, X_test_temp, y_test_temp = get_xy(test, "bike_count", x_labels=["temp"])

temp_reg = LinearRegression()
temp_reg.fit(X_train_temp, y_train_temp)

temp_reg.score(X_test_temp, y_test_temp)

dir_except_checkpoints('regression')

plt.scatter(X_train_temp, y_train_temp, label="Data", color="blue")
x = tf.linspace(-20, 40, 100)
plt.plot(x, temp_reg.predict(np.array(x).reshape(-1, 1)), label="Fit", color="red", linewidth=3)
plt.legend()
plt.title("Bikes vs Temp")
plt.ylabel("Number of bikes")
plt.xlabel("Temp")
plt.savefig(os.path.join('regression', 'bikes_vs_temp.png'))
plt.clf()

# Multiple Linear Regression
train, val, test = np.split(df.sample(frac=1), [int(0.6*len(df)), int(0.8*len(df))])
_, X_train_all, y_train_all = get_xy(train, "bike_count", x_labels=df.columns[1:])
_, X_val_all, y_val_all = get_xy(val, "bike_count", x_labels=df.columns[1:])
_, X_test_all, y_test_all = get_xy(test, "bike_count", x_labels=df.columns[1:])

all_reg = LinearRegression()
all_reg.fit(X_train_all, y_train_all)

all_reg.score(X_test_all, y_test_all)

# Regression with Neural Net
dir_except_checkpoints('neural_net')

def plot_loss(history, directory, file_name):
    plt.plot(history.history['loss'], label='loss')
    plt.plot(history.history['val_loss'], label='val_loss')
    plt.xlabel('Epoch')
    plt.ylabel('MSE')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(directory, f'{file_name}.png'))
    plt.clf()

temp_normalizer = tf.keras.layers.Normalization(input_shape=(1,), axis=None)
temp_normalizer.adapt(X_train_temp.reshape(-1))

temp_nn_model = tf.keras.Sequential([
    temp_normalizer,
    tf.keras.layers.Dense(1)
])

temp_nn_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.1), loss='mean_squared_error')

history_temp_nn_model = temp_nn_model.fit(
    X_train_temp.reshape(-1), y_train_temp,
    epochs=1000,
    validation_data=(X_val_temp, y_val_temp),
    # If you do not like to see training in console
    # verbose=0
    verbose=0
)

plot_loss(history_temp_nn_model, 'neural_net', 'temp_nn_model_history_loss_val_loss')

plt.scatter(X_train_temp, y_train_temp, label="Data", color="blue")
x = tf.linspace(-20, 40, 100)
plt.plot(x, temp_nn_model.predict(np.array(x).reshape(-1, 1)), label="Fit", color="red", linewidth=3)
plt.legend()
plt.title("Bikes vs Temp")
plt.ylabel("Number of bikes")
plt.xlabel("Temp")
plt.savefig(os.path.join('neural_net', 'bikes_vs_temp_temp_nn_model.png'))
plt.clf()

# Neural Net
temp_normalizer = tf.keras.layers.Normalization(input_shape=(1,), axis=None)
temp_normalizer.adapt(X_train_temp.reshape(-1))

nn_model = tf.keras.Sequential([
    temp_normalizer,
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(1)
])

nn_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss='mean_squared_error')

history_nn_model = nn_model.fit(
    X_train_temp, y_train_temp,
    validation_data=(X_val_temp, y_val_temp),
    epochs=100,
    # If you do not like to see training in console
    # verbose=0
    verbose=0
)

plot_loss(history_nn_model, 'neural_net', 'nn_model_history_loss_val_loss')

plt.scatter(X_train_temp, y_train_temp, label="Data", color="blue")
x = tf.linspace(-20, 40, 100)
plt.plot(x, nn_model.predict(np.array(x).reshape(-1, 1)), label="Fit", color="red", linewidth=3)
plt.legend()
plt.title("Bikes vs Temp")
plt.ylabel("Number of bikes")
plt.xlabel("Temp")
plt.savefig(os.path.join('neural_net', 'bikes_vs_temp_nn_model.png'))
plt.clf()

all_normalizer = tf.keras.layers.Normalization(input_shape=(6,), axis=-1)
all_normalizer.adapt(X_train_all)

nn_model_all = tf.keras.Sequential([
    all_normalizer,
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(1)
])

nn_model_all.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss='mean_squared_error')

history_nn_model_all = nn_model_all.fit(
    X_train_all, y_train_all,
    validation_data=(X_val_all, y_val_all),
    epochs=100,
    # If you do not like to see training in console
    # verbose=0
    verbose=0
)

plot_loss(history_nn_model_all, 'neural_net', 'nn_model_all_history_loss_val_loss')

# Calculate the Mean Squared Error for both linear regression and neural net
y_pred_lr = all_reg.predict(X_test_all)
y_pred_nn = nn_model_all.predict(X_test_all)

def MSE(y_pred, y_real):
    return (np.square(y_pred - y_real)).mean()

print("MSE Linear Regression:", MSE(y_pred_lr, y_test_all))
print("MSE Neural Net:", MSE(y_pred_nn, y_test_all))

dir_except_checkpoints('predictions')

ax = plt.axes(aspect="equal")
plt.scatter(y_test_all, y_pred_lr, label="Lin Reg Preds")
plt.scatter(y_test_all, y_pred_nn, label="NN Preds")
plt.xlabel("True Values")
plt.ylabel("Predictions")
lims = [0, 1800]
plt.xlim(lims)
plt.ylim(lims)
_ = plt.plot(lims, lims, c="red")
plt.legend()
plt.savefig(os.path.join('predictions', 'lin_reg_nn_pred.png'))
plt.clf() 