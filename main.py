from preprocessing_eda import fetch_API_data, preprocess, coin_selection
from prepare_data import prepare_data, splitting_train_test_data
from models import lstm_model, train_lstm_model, lstm_model_loss
from predictions import test_prediction, calculate_rmse, plot_test_predictions, future_prediction

# Fetch data
data = fetch_API_data()

# Preprocess data
preprocessed_data = preprocess(data)

# Select a coin
selected_coin_data = coin_selection(preprocessed_data)

# Prepare and split data
train_data, test_data, scaler = prepare_data(selected_coin_data)
x_train, y_train, x_test, y_test = splitting_train_test_data(train_data, test_data)

# Train LSTM model
input_shape = (x_train.shape[1], x_train.shape[2])
model = lstm_model(input_shape)
history = train_lstm_model(model, x_train, y_train)

# Plot losses
lstm_model_loss(history)

# Test predictions
predictions = test_prediction(model, x_test, scaler)
rmse = calculate_rmse(y_test, predictions, scaler)
print(f"Root Mean Squared Error: {rmse}")

# Plot test predictionsET
plot_test_predictions(predictions, y_test, scaler)

# Make future predictions
N = 30  # number of days for future predictions
time_steps = 60
future_price_df = future_prediction(model,selected_coin_data, test_data, time_steps, scaler, N)
print(future_price_df)
