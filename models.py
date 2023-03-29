import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from keras.models import Sequential
from keras.callbacks import EarlyStopping
from keras.layers import Dense, LSTM, Dropout

def lstm_model(input_shape, n_cols=1):
    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True, input_shape=input_shape))
    model.add(Dropout(0.2))
    model.add(LSTM(units=50, return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(units=50))
    model.add(Dropout(0.2))
    model.add(Dense(units=1))
    
    model.compile(optimizer = 'adam', loss= 'mse', metrics = 'mean_absolute_error')
    return model 

def train_lstm_model(model, x_train, y_train, epochs=100, batch_size=32, verbose=1):
    history = model.fit(x_train, y_train, epochs = epochs, batch_size= batch_size, verbose=verbose)
    return history

def lstm_model_loss(history):
    # model eveluation
    plt.figure(figsize=(12,8))
    plt.plot(history.history['loss'])
    plt.plot(history.history['mean_absolute_error'])
    plt.legend(['Mean Squared Error', 'Mean Absolute Error'])
    plt.title("Losses")
    plt.xlabel("epochs")
    plt.ylabel("loss")
    plt.show()