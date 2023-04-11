import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import RandomOverSampler

from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report

from sklearn.naive_bayes import GaussianNB

from sklearn.linear_model import LogisticRegression

from sklearn.svm import SVC

import tensorflow as tf
import os
from utils.helper_functions import dir_except_checkpoints

# Dataset:
# Dua, D. and Graff, C. (2019). UCI Machine Learning Repository [https://archive.ics.uci.edu/ml]. Irvine, CA: University of California,
# School of Information and Computer Science.

# Donated by:
# P. Savicky
# Institute of Computer Sciense, AS of CR
# Czech Republic
# savicky '@' cs.cas.cz

cols = ["fLength", "fWidth", "fSize", "fConc", "fConc1", "fAsym", "fM3Long", "fM3Trans", "fAlpha", "fDist", "class"]
df = pd.read_csv("./data/magic04.data", names=cols)

df["class"] = (df["class"] == 'g').astype(int)

dir_except_checkpoints('probability')

for label in cols[:-1]:
    plt.hist(df[df["class"]==1][label], color='blue', label='gamma', alpha=0.7, density=True)
    plt.hist(df[df["class"]==0][label], color='red', label='hadron', alpha=0.7, density=True)
    plt.title(label)
    plt.ylabel("Probability")
    plt.xlabel(label)
    plt.legend()
    plt.savefig(os.path.join('probability', f'{label}.png'))
    plt.clf()

for label in cols[:-1]:
    plt.hist(df[df["class"]==1][label], color='blue', label='gamma', alpha=0.7, density=True)
    plt.hist(df[df["class"]==0][label], color='red', label='hadron', alpha=0.7, density=True)
    plt.title(label)
    plt.ylabel("Probability")
    plt.xlabel(label)
    plt.legend()
    plt.savefig(os.path.join('probability', f'{label}.png'))
    plt.clf()

    
# Train, Validation, Test Dataset
train, valid, test = np.split(df.sample(frac=1), [int(0.6*len(df)), int(0.8*len(df))])

def scale_dataset(dataframe, oversample=False):
  if not isinstance(dataframe, pd.DataFrame):
    dataframe = pd.DataFrame(dataframe)

  X = dataframe[dataframe.columns[:-1]].values
  y = dataframe[dataframe.columns[-1]].values

  scaler = StandardScaler()
  X = scaler.fit_transform(X)

  if oversample:
    ros = RandomOverSampler()
    X, y = ros.fit_resample(X, y)

  data = np.hstack((X, np.reshape(y, (-1, 1))))

  return data, X, y

train, X_train, y_train = scale_dataset(train, oversample=True)
valid, X_valid, y_valid = scale_dataset(valid, oversample=False)
test, X_test, y_test = scale_dataset(test, oversample=False)

# KNN
knn_model = KNeighborsClassifier(n_neighbors = 5)
knn_model.fit(X_train, y_train)

y_pred = knn_model.predict(X_test)

y_pred

print(classification_report(y_test, y_pred))

# Naive Bayes
nb_model = GaussianNB()
nb_model = nb_model.fit(X_train, y_train)

y_pred = nb_model.predict(X_test)
print(classification_report(y_test, y_pred))

# Logistic Regression
lg_model = LogisticRegression()
lg_model = lg_model.fit(X_train, y_train)

y_pred = lg_model.predict(X_test)
print(classification_report(y_test, y_pred))

# SVM
svm_model = SVC()
svm_model = svm_model.fit(X_train, y_train)

y_pred = svm_model.predict(X_test)
print(classification_report(y_test, y_pred))

# Neural Net
dir_except_checkpoints('history')

def plot_history(history, plot_num):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    ax1.plot(history.history['loss'], label='loss')
    ax1.plot(history.history['val_loss'], label='val_loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Binary crossentropy')
    ax1.grid(True)

    ax2.plot(history.history['accuracy'], label='accuracy')
    ax2.plot(history.history['val_accuracy'], label='val_accuracy')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.grid(True)

    plt.savefig(os.path.join('history', f'plot_{plot_num}.png'))
    plt.clf()

def train_model(X_train, y_train, num_nodes, dropout_prob, lr, batch_size, epochs):
    nn_model = tf.keras.Sequential([
        tf.keras.layers.Dense(num_nodes, activation='relu', input_shape=(10,)),
        tf.keras.layers.Dropout(dropout_prob),
        tf.keras.layers.Dense(num_nodes, activation='relu'),
        tf.keras.layers.Dropout(dropout_prob),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])

    nn_model.compile(optimizer=tf.keras.optimizers.Adam(lr), loss='binary_crossentropy',
                  metrics=['accuracy'])
  
    history = nn_model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_split=0.2, verbose=0)

    return nn_model, history

least_val_loss = float('inf')
least_loss_model = None
epochs=100

param_list = [(num_nodes, dropout_prob, lr, batch_size) for num_nodes in [16, 32, 64] 
              for dropout_prob in [0, 0.2] for lr in [0.01, 0.005, 0.001] 
              for batch_size in [32, 64, 128]]

for i, (num_nodes, dropout_prob, lr, batch_size) in enumerate(param_list):
    print(f"nodes {num_nodes}, dropout {dropout_prob}, learning rate {lr}, batch size {batch_size}")
    model, history = train_model(X_train, y_train, num_nodes, dropout_prob, lr, batch_size, epochs)
    plot_history(history, i)
    val_losses = model.evaluate(X_valid, y_valid)
    val_loss = sum(val_losses) / len(val_losses)
    if val_loss < least_val_loss:
        least_val_loss = val_loss
        least_loss_model = model

y_pred = least_loss_model.predict(X_test)
y_pred = (y_pred > 0.5).astype(int).reshape(-1,)

y_pred

print(classification_report(y_test, y_pred))