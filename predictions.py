import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def test_prediction(model, x_test, scaler):
    prediction = model.predict(x_test)
    prediction = scaler.inverse_transform(prediction)
    return prediction

def calculate_rmse(y_test, prediction, scaler):
    y_test = scaler.inverse_transform(y_test)
    rmse = np.sqrt(np.mean(y_test - prediction)**2).round(2)
    return rmse

def plot_test_predictions(prediction, y_test, scaler):
    y_test = scaler.inverse_transform(y_test)
    preds_acts = pd.DataFrame(data={'Predictions' : prediction.flatten(), 'Actuals' : y_test.flatten()})
    plt.figure(figsize=(16,6))
    plt.plot(preds_acts['Predictions'])
    plt.plot(preds_acts['Actuals'])
    plt.legend(['Predictions','Actuals'])
    plt.show()

def future_prediction(model, selected_coin_data, scaled_data, time_steps, scaler, N):
    last_date = selected_coin_data['date'].max()
    future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=N, closed='left')

    future_preds = []
    last_60_days = scaled_data[-time_steps:]

    for i in range(N):
        input_data = last_60_days[-time_steps:]
        input_data = np.array(input_data)
        input_data = np.reshape(input_data, (1, time_steps, 1))

        predicted_price = model.predict(input_data)
        future_preds.append(predicted_price[0][0])

        last_60_days = np.append(last_60_days, predicted_price, axis=0)

    future_preds = scaler.inverse_transform(np.array(future_preds).reshape(-1, 1))
    future_price_df = pd.DataFrame(future_preds, columns=['Predicted Close Price'], index=future_dates)
    return future_price_df
