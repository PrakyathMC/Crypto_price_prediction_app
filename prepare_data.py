import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler


def prepare_data(selected_coin_data,column='close',time_steps=60,train_size=0.75):
    dataset = selected_coin_data[column]
    dataset = pd.DataFrame(dataset)
    data = dataset.values

    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(np.array(data))
    train_size = int(len(data) * train_size)
    test_size = len(data) - train_size

    train_data = scaled_data[0:train_size, :]
    test_data = scaled_data[train_size - time_steps:, :]

    return train_data, test_data, scaler

def splitting_train_test_data(train_data, test_data, time_steps=60):
    x_train, y_train, x_test, y_test = [], [], [], []

    for i in range(time_steps, len(train_data)):
        x_train.append(train_data[i - time_steps:i])
        y_train.append(train_data[i])

    for i in range(time_steps, len(test_data)):
        x_test.append(test_data[i - time_steps:i])
        y_test.append(test_data[i])

    x_train, y_train = np.array(x_train), np.array(y_train)
    x_test, y_test = np.array(x_test), np.array(y_test)

    return x_train, y_train, x_test, y_test
