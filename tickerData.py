from pathlib import Path

import numpy as np
import pandas as pd
import run_model_3 as rm3
import yfinance as yf
from tensorflow.keras.models import model_from_json


def getTickerPriceData(tickers,period='5d',interval='1d'):
    #Getting Ticker Price Data (Open,High,Close,etc)
    ticker_df = yf.download(tickers=tickers,period=period,interval=interval)
    return ticker_df

def calculate_macd(data,short_window=8,long_window=22,verbose=0):
    EMA_short = data.ewm(halflife=short_window).mean()
    EMA_long = data.ewm(halflife=long_window).mean()
    macd = EMA_short - EMA_long

    macd_signal = np.where(macd > 0, 1.0, 0.0)
    if verbose:
        print(f'The short window is {short_window} & the long window is {long_window}!')
    return macd,macd_signal

def EWMA(data, ndays):
    EMA = pd.Series(data['Close'].ewm(span=ndays, min_periods=ndays - 1).mean(),
                    name='EWMA_' + str(ndays))
    data = data.join(EMA)
    return data

def computeRSI(data, time_window):
    diff = data.diff(1).dropna()  # diff in one field(one day)

    # this preservers dimensions off diff values
    up_chg = 0 * diff
    down_chg = 0 * diff

    # up change is equal to the positive difference, otherwise equal to zero
    up_chg[diff > 0] = diff[diff > 0]

    # down change is equal to negative deifference, otherwise equal to zero
    down_chg[diff < 0] = diff[diff < 0]

    # we set com=time_window-1 so we get decay alpha=1/time_window
    up_chg_avg = up_chg.ewm(com=time_window - 1, min_periods=time_window).mean()
    down_chg_avg = down_chg.ewm(com=time_window - 1, min_periods=time_window).mean()

    rs = abs(up_chg_avg / down_chg_avg)
    rsi = 100 - 100 / (1 + rs)
    return rsi

def calculate_BollingerBands(data,long_window=21):
    ma_long = data.rolling(window=long_window).mean()
    long_std = data.rolling(window=long_window).std()

    upper = ma_long + (long_std * 2)
    lower = ma_long - (long_std * 2)

    return upper,lower

def makeTickerDfSignals(ticker_data_df,interval='1d',short_window=9,long_window=21,initial_capital=10000.00,shares=500):
    #Add computational signals to the ticker dataframe

    # Day Length Trade Intervals:

    signals_df = ticker_data_df.loc[:,['Close']].copy()

    # Calculate Daily Returns
    signals_df['Daily_Return'] = signals_df['Close'].dropna().pct_change()
    signals_df.dropna(inplace=True)

    # Generate the short and long moving averages (short window and long window days, respectively)
    signals_df['SMA%s'%short_window] = signals_df['Close'].rolling(window=short_window).mean()
    signals_df['SMA%s'%long_window] = signals_df['Close'].rolling(window=long_window).mean()

    # Initialize the new `Signal` column
    signals_df['Signal'] = 0.0

    signals_df.dropna(inplace=True)
    # Generate the trading signal (1 or 0) to when the short window is less than the long
    # Note: Use 1 when the SMA50 is less than SMA100 and 0 for when it is not.
    signals_df['Signal'][short_window:] = np.where(
        signals_df["SMA%s" % short_window][short_window:] > signals_df["SMA%s" % long_window][short_window:], 1.0,
        0.0)

    # Construct a `Fast` and `Slow` Exponential Moving Average from short and long window close prices, respectively
    signals_df['fast_close_%s'%short_window] = signals_df['Close'].ewm(halflife=short_window).mean()
    signals_df['slow_close_%s'%long_window] = signals_df['Close'].ewm(halflife=long_window).mean()

    # Construct a `Fast` and `Slow` Exponential Moving Average from short and long windows, respectively
    signals_df['fast_vol'] = signals_df['Daily_Return'].ewm(halflife=short_window).std()
    signals_df['slow_vol'] = signals_df['Daily_Return'].ewm(halflife=long_window).std()

    # Calculate the points in time at which a position should be taken, 1 or -1
    signals_df["Entry/Exit"] = signals_df["Signal"].diff()

    # RSI Indicator
    signals_df['RSI'] = computeRSI(signals_df['Close'], time_window=14)
    signals_df['RSI_Upper_Lim'] = 70
    signals_df['RSI_Lower_Lim'] = 30

    # MACD Indicator
    macd, macdsignal = calculate_macd(signals_df['Close'],short_window=short_window,long_window=long_window)
    signals_df['MACD'] = macd
    signals_df['MACD_Sig'] = macdsignal

    #Bollinger Bands
    upper_band,lower_band = calculate_BollingerBands(signals_df['Close'], long_window=long_window)
    signals_df['Upper_Band'] = upper_band
    signals_df['Lower_Band'] = lower_band

    signals_df.dropna(inplace=True)

    return signals_df


def execute_backtest(data_df,initial_capital=10000.00,shares=500):
    # Take the number of shares positions where the dual moving average crossover is 1 (SMA50 is greater than SMA100)
    data_df["Position"] = shares * data_df["Signal"]

    # Find the points in time where a 500 share position is bought or sold
    data_df["Entry/Exit Position"] = data_df["Position"].diff()

    # Multiply share price by entry/exit positions and get the cumulatively sum
    data_df["Portfolio Holdings"] = (data_df["Close"] * data_df["Entry/Exit Position"].cumsum())

    # Subtract the initial capital by the portfolio holdings to get the amount of liquid cash in the portfolio
    data_df["Portfolio Cash"] = (initial_capital - (data_df["Close"] * data_df["Entry/Exit Position"]).cumsum())

    # Get the total portfolio value by adding the cash amount by the portfolio holdings (or investments)
    data_df["Portfolio Total"] = (data_df["Portfolio Cash"] + data_df["Portfolio Holdings"])

    # Calculate the portfolio daily returns
    data_df["Portfolio Daily Returns"] = data_df["Portfolio Total"].pct_change()

    # Calculate the cumulative returns
    data_df["Portfolio Cumulative Returns"] = (1 + data_df["Portfolio Daily Returns"]).cumprod() - 1

    # Make some predictions with the loaded model
    model = load_model()

    #data_split = rm3.load_data(data_df, n_steps=50, scale=True, shuffle=True, lookup_step=1, split_by_date=True,
                #test_size=0.2, feature_columns=['Adj Close', 'Volume', 'Open', 'High', 'Low'])


    # Final Dataframe - running predictions and getting predicted prices & option chains
    #final_df,recommendation, predicted_price, strike_price_call, strike_price_put = rm3.recommendation(model, data_split)


    #return final_df,recommendation, predicted_price, strike_price_call, strike_price_put
    return data_df

def load_model(model_file="model_3_15day.json",weights_file="model_3_15day.h5"):
    # load json and create model
    file_path = Path(model_file)
    with open(file_path, "r") as json_file:
        model_json = json_file.read()
    loaded_model = model_from_json(model_json)

    # load weights into new model
    file_path = "model_3_15day.h5"
    loaded_model.load_weights(weights_file)

    return loaded_model