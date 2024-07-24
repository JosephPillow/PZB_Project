import os
import pandas as pd
import numpy as np
import talib as ta
import sqlite3
from binance.client import Client
from binance.enums import *
from cryptography.hazmat.primitives import serialization
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import time

# Load private and public keys
with open("test-prv-key.pem", "rb") as key_file:
    private_key = serialization.load_pem_private_key(
        key_file.read(),
        password=None,
    )

with open("test-pub-key.pem", "rb") as key_file:
    public_key = serialization.load_pem_public_key(
        key_file.read()
    )

# Initialize Binance client
API_KEY = 'd0r5RAa8BwjyG0wpsWgpjrUZ2vCObBU0Aqo6rrMlK5ALOpcUy6gSZP4xPbXEdTDk'
API_SECRET = 'DDwKoq5qk3cYxoaAyc5hqDk6Kore07qt0nKvun5ucEJixHzOHOsCeAQPGw6OJXB8'
client = Client(API_KEY, API_SECRET, testnet=True)

# Define cryptocurrencies to analyze
coins = ['BTCUSDT', 'ETHUSDT', 'BNBUSDT', 'ADAUSDT', 'XRPUSDT', 'DOGEUSDT', 'DOTUSDT', 'UNIUSDT', 'LTCUSDT', 'LINKUSDT']

def fetch_and_analyze(coin):
    bars = client.get_historical_klines(coin, Client.KLINE_INTERVAL_1DAY, '365 day ago UTC')
    df = pd.DataFrame(bars, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume', 'close_time', 'quote_asset_volume', 'number_of_trades', 'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df.set_index('timestamp', inplace=True)

    # Technical analysis indicators
    df['SMA50'] = ta.SMA(df['close'].astype(float), timeperiod=50)
    df['SMA200'] = ta.SMA(df['close'].astype(float), timeperiod=200)
    df['EMA12'] = ta.EMA(df['close'].astype(float), timeperiod=12)
    df['EMA26'] = ta.EMA(df['close'].astype(float), timeperiod=26)
    df['RSI'] = ta.RSI(df['close'].astype(float), timeperiod=14)
    df['MACD'], df['MACD_signal'], df['MACD_hist'] = ta.MACD(df['close'].astype(float), fastperiod=12, slowperiod=26, signalperiod=9)
    df['upper_band'], df['middle_band'], df['lower_band'] = ta.BBANDS(df['close'].astype(float), timeperiod=20)
    df['ADX'] = ta.ADX(df['high'].astype(float), df['low'].astype(float), df['close'].astype(float), timeperiod=14)
    df['stoch_k'], df['stoch_d'] = ta.STOCH(df['high'].astype(float), df['low'].astype(float), df['close'].astype(float), fastk_period=14, slowk_period=3, slowd_period=3)

    df['target'] = np.where(df['close'].shift(-1) > df['close'], 1, 0)

    return df

# Save data to SQLite database
conn = sqlite3.connect('crypto_analysis.db')
c = conn.cursor()

# Create tables
for coin in coins:
    table_name = coin
    c.execute(f"""
        CREATE TABLE IF NOT EXISTS {table_name} (
            timestamp TEXT,
            open REAL,
            high REAL,
            low REAL,
            close REAL,
            volume REAL,
            SMA50 REAL,
            SMA200 REAL,
            EMA12 REAL,
            EMA26 REAL,
            RSI REAL,
            MACD REAL,
            MACD_signal REAL,
            MACD_hist REAL,
            upper_band REAL,
            middle_band REAL,
            lower_band REAL,
            ADX REAL,
            stoch_k REAL,
            stoch_d REAL
        )
    """)

# Fetch and save data
for coin in coins:
    df = fetch_and_analyze(coin)
    table_name = coin
    df.to_sql(table_name, conn, if_exists='replace', index_label='timestamp')

conn.commit()

def fetch_wallet_balance():
    balance = client.get_account()
    total_balance = {}
    for asset in balance['balances']:
        total_balance[asset['asset']] = float(asset['free']) + float(asset['locked'])
    return total_balance

def prepare_data(coin):
    table_name = coin
    df = pd.read_sql(f"SELECT * FROM {table_name}", conn, parse_dates=['timestamp'])

    # Convert object columns to float
    for col in df.columns:
        if df[col].dtype == 'object':
            try:
                df[col] = df[col].astype(float)
            except ValueError:
                print(f"Column '{col}' could not be converted to float.")

    # Clean NaN and infinite values
    df.dropna(inplace=True)
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(inplace=True)

    features = ['SMA50', 'SMA200', 'EMA12', 'EMA26', 'RSI', 'MACD', 'MACD_signal', 'MACD_hist', 'upper_band',
                'middle_band', 'lower_band', 'ADX', 'stoch_k', 'stoch_d']
    X = df[features]
    y = df['target']

    # Scale data
    mms = MinMaxScaler()
    mms.fit(X)
    X_scaled = mms.transform(X)

    return X_scaled, y, mms

# Train and test models
models = {}
scalers = {}
wallets = {}
for coin in coins:
    X, y, scaler = prepare_data(coin)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = LogisticRegression(max_iter=500)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"{coin} Model Accuracy: {accuracy:.2f}")
    models[coin] = model
    scalers[coin] = scaler

    # Initialize wallet balance tracking
    wallets[coin] = 0

# Execute trades and check for stop-loss
def execute_trades(coin):
    df = fetch_and_analyze(coin)
    table_name = coin
    df.to_sql(table_name, conn, if_exists='replace', index_label='timestamp')

    # Get latest data
    latest_data = df.iloc[-1]
    features = ['SMA50', 'SMA200', 'EMA12', 'EMA26', 'RSI', 'MACD', 'MACD_signal', 'MACD_hist',
                'upper_band', 'middle_band', 'lower_band', 'ADX', 'stoch_k', 'stoch_d']

    latest_data = latest_data[features].values.reshape(1, -1)
    scaler = scalers[coin]
    latest_data_scaled = scaler.transform(latest_data)

    model = models[coin]
    prediction = model.predict(latest_data_scaled)[0]

    # Wallet balance
    balance = fetch_wallet_balance()
    wallet_coin_balance = balance.get(coin[:-4], 0)  # Get balance for the coin

    quantity = 0.01  # Example quantity, use with caution in real trading
    if prediction == 1 and wallet_coin_balance > 0:
        print(f"{coin} - Buy Signal")
        # Place buy order
        order = client.futures_create_order(
            symbol=coin,
            side=SIDE_BUY,
            type=ORDER_TYPE_MARKET,
            quantity=quantity
        )
        print(f"Buy order placed: {order}")
        wallets[coin] += quantity
    elif prediction == 0 and wallets[coin] > 0:
        print(f"{coin} - Sell Signal")
        # Place sell order
        order = client.futures_create_order(
            symbol=coin,
            side=SIDE_SELL,
            type=ORDER_TYPE_MARKET,
            quantity=quantity
        )
        print(f"Sell order placed: {order}")
        wallets[coin] -= quantity

def check_for_stop_loss(coin):
    df = fetch_and_analyze(coin)
    latest_close = df['close'].iloc[-1]
    purchase_price = df['close'].iloc[-2]  # Using the previous close as the purchase price

    # Check for a >30% drop
    if (purchase_price - latest_close) / purchase_price > 0.30:
        print(f"{coin} - Stop-Loss: Coin dropped more than 30%. Sell!")
        # Place stop-loss order
        quantity = 0.01  # Example quantity, use with caution in real trading
        order = client.futures_create_order(
            symbol=coin,
            side=SIDE_SELL,
            type=ORDER_TYPE_MARKET,
            quantity=quantity
        )
        print(f"Stop-loss sell order placed: {order}")

# Continuous bot loop
while True:
    for coin in coins:
        execute_trades(coin)
        check_for_stop_loss(coin)

    # Wait for a specified period (e.g., 5 minutes = 300 seconds)
    time.sleep(300)
