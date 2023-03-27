import matplotlib
import pandas as pd
import pandas_datareader as pdr
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt 
from sklearn.preprocessing import MinMaxScaler

# Function to fetch data from API
def fetch_API_data():
    symbols = ['BTCUSD', 'ETHUSD', 'LTCUSD','BCHUSD','XRPUSD','LINKUSD','ADAUSD','DOTUSD','UNIUSD','DOGEUSD','ETCUSD','MATICUSD',
    'BSVUSD','FILUSD','ATOMUSD','XLMUSD','AAVEUSD','CAKEUSD','SUSHIUSD','MKRUSD','AVAXUSD']
    data = []

    # Loop through the list of symbols and fetch the data for each symbol
    for symbol in symbols:
        df = pdr.get_data_tiingo(symbol, start='2020-01-01', end='2022-12-25', api_key='d8c623538ff22efb5b7342a198b63b1e430e93f3')
        data.append(df)
        print(f'{symbol} data fetched and saved.')

    # Concatenate the dataframes in di into a single dataframe
    df = pd.concat(data, keys=symbols)

    # Print the first few rows of the dataframe
    print(df.head())
    return df

def preprocess(df):
    df.reset_index(inplace=True)
    #dropping values
    print(df['level_0'].unique())
    print("")
    print(df['symbol'].unique())
    df.drop('level_0',axis=1,inplace=True)
    df = df.drop(['adjClose','adjHigh','adjLow','adjOpen','adjVolume','divCash','splitFactor'], axis=1)
    df.head(2)
    df['date'] = pd.to_datetime(df['date']).dt.date
    return df

def missing_values(df):
    matplotlib.rcParams['figure.figsize'] = (8,6)
    sns.heatmap(df.isnull(),cbar=False,cmap='viridis')

def countplot(df):
    matplotlib.rcParams['figure.figsize'] = (22,9)
    sns.countplot(x="symbol", data=df)

def pie_chart(df):
    df.columns
    coin_names = df['symbol'].value_counts().index
    coin_names
    coin_values = df['symbol'].value_counts().values
    coin_values
    matplotlib.rcParams['figure.figsize'] = (13,8)
    plt.pie(x=coin_values,labels=coin_names,autopct='%1.2f%%')
    plt.show()

def coin_selection(df):
    coins = input("Please select a coin out of the 21 coins available:\n\n"
            "BTCUSD - for Bitcoin\n" 
            "ETHUSD - for Etherium\n"
            "LTCUSD - for Litecoin\n"
            "BCHUSD - for Bitcoin cash\n"
            "XRPUSD - for XRP\n"
            "LINKUSD - for Chainlink\n"
            "ADAUSD - for Cardano\n"
            "DOTUSD - for Polkadot\n"
            "UNIUSD - for Uniswap\n"
            "DOGEUSD - Dogecoin\n"
            "ETCUSD - for Ethereum Classic\n"
            "MATICUSD - for Polygon (formerly Matic Network)\n"
            "BSVUSD - for Bitcoin SV\n"
            "FILUSD - for Filecoin\n"
            "ATOMUSD - for Cosmos\n"
            "XLMUSD - for Stellar\n"
            "AAVEUSD - for Aave\n"
            "CAKEUSD - for PancakeSwap\n"
            "SUSHIUSD - for SushiSwap\n"
            "MKRUSD - for Maker\n"
            "AVAXUSD - for Avalanche\n\n")

    preferred_coin = df[df['symbol']==coins]
    print(preferred_coin.head())
    print(preferred_coin.tail())
    print(preferred_coin.shape)
    
    df[df['symbol']==coins].max()
    
    # finding the maximum and minimum values in close

    max_val = preferred_coin['close'].max()
    print(f'The maximum value of {coins} from {max_val} \n')

    min_val = preferred_coin['close'].min()
    print(f'The minimum value of {coins} from {min_val} ')
    return preferred_coin

def close_vs_date(preferred_coin):
    
    fig = plt.figure(figsize=(10,6))
    axes= fig.add_axes([0.1, 0.1, 0.8, 0.8])
    axes.plot(preferred_coin['date'],preferred_coin['close'], 'b')
    axes.set_xlabel("Date")
    axes.set_ylabel(f"{preferred_coin['symbol'].iloc[0]} Close Price")
    plt.show()

def histogram(preferred_coin):
    total_values = preferred_coin['close'].count()
    print(f"The total number of values in {preferred_coin['symbol'].iloc[0]} are: {total_values}")
    import math
    square_root_value = math.sqrt(total_values)
    print(f'The square root of {total_values} is: {square_root_value}')

    plt.hist(preferred_coin['close'], bins=33 ,color='b')
    plt.xlabel('Date')
    plt.ylabel(f"{preferred_coin['symbol'].iloc[0]} close price")
    plt.show()

def open_close_coin(preferred_coin):
    # checking the opening and closing prices of bitcoin

    plt.figure(figsize=(15,6))
    preferred_coin['open'].plot()
    preferred_coin['close'].plot()
    plt.title(f"Opening and closing price of {preferred_coin['symbol'].iloc[0]}")
    plt.legend(['Open price','close price'])
    plt.show()
    
    # high/low prices
    #high and low prices 

    plt.figure(figsize=(15,6))
    preferred_coin['high'].plot()
    preferred_coin['low'].plot()
    plt.title(f"high and low price of {preferred_coin['symbol'].iloc[0]}")
    plt.legend(['high price','low price'])
    plt.show()



# Outputs:-

if __name__ == "__main__":

    # Fetch data
    data = fetch_API_data()

    #Preprocessed outputs
    preprocessed_data = preprocess(data)

    # Checking for missing values
    missing_values(preprocessed_data)

    #Display countplot
    countplot(preprocessed_data)

    #Display pie chart
    pie_chart(preprocessed_data)

    # User selected coin & its data
    selected_coin_data = coin_selection(preprocessed_data)

    #Close vs date plot
    close_vs_date(selected_coin_data)

    # Display histogram
    histogram(selected_coin_data)

    # Open vs Close plot
    open_close_coin(selected_coin_data)