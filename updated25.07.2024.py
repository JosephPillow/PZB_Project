import os
import ccxt
import pandas as pd
import numpy as np
import talib as ta
import sqlite3
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Input
from tensorflow.keras.optimizers import Adam
import time

# API Anahtarlarını Güvenli Bir Şekilde Alın
api_key = '8lrngwZQgIs9tWgC5rgxdgTNmkygxIeQN6qYAtcrHH1y36PAMWI7IvlEZUWDn3aX'
api_secret = 'KLmsyE9sa0ay5Y48SoLcIYwokJA9XOT2JtvkMxSD9KUezhBhS4MiAdcUj6Vd3dAN'

# Binance API'yi Ayarlama
exchange = ccxt.binance({
    'apiKey': api_key,
    'secret': api_secret,
    'urls': {
        'api': {
            'public': 'https://testnet.binance.vision/api/v3',
            'private': 'https://testnet.binance.vision/api/v3',
        },
    },
})

# Zaman Farkı Düzeltmesini Etkinleştirin
exchange.options['adjust_for_time_difference'] = True
exchange.set_sandbox_mode(True)

# İnceleme yapılacak kripto paralar
coins = ['BTC/USDT', 'ETH/USDT', 'BNB/USDT', 'ADA/USDT', 'XRP/USDT', 'DOGE/USDT', 'DOT/USDT', 'UNI/USDT', 'LTC/USDT',
         'LINK/USDT']


def fetch_and_analyze(coin):
    try:
        bars = exchange.fetch_ohlcv(coin, timeframe='1m', limit=365)
    except Exception as e:
        print(f"Error fetching data for {coin}: {e}")
        return pd.DataFrame()  # Boş DataFrame döndür

    df = pd.DataFrame(bars, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df.set_index('timestamp', inplace=True)

    # Teknik analiz göstergeleri
    df['SMA50'] = ta.SMA(df['close'], timeperiod=50)
    df['SMA200'] = ta.SMA(df['close'], timeperiod=200)
    df['EMA12'] = ta.EMA(df['close'], timeperiod=12)
    df['EMA26'] = ta.EMA(df['close'], timeperiod=26)
    df['RSI'] = ta.RSI(df['close'], timeperiod=14)
    df['MACD'], df['MACD_signal'], df['MACD_hist'] = ta.MACD(df['close'], fastperiod=12, slowperiod=26, signalperiod=9)
    df['upper_band'], df['middle_band'], df['lower_band'] = ta.BBANDS(df['close'], timeperiod=20)
    df['ADX'] = ta.ADX(df['high'], df['low'], df['close'], timeperiod=14)
    df['stoch_k'], df['stoch_d'] = ta.STOCH(df['high'], df['low'], df['close'], fastk_period=14, slowk_period=3,
                                            slowd_period=3)

    df['target'] = np.where(df['close'].shift(-1) > df['close'], 1, 0)

    return df

def fetch_wallet_balance():
    try:
        balance = exchange.fetch_balance()
        return balance
    except ccxt.NetworkError as e:
        print(f"Network error: {e}")
    except ccxt.ExchangeError as e:
        print(f"Exchange error: {e}")
    except Exception as e:
        print(f"Error fetching wallet balance: {e}")
    return {}
def prepare_data(df):
    # Özellikler ve hedef değişkenin ayrılması
    features = ['SMA50', 'SMA200', 'EMA12', 'EMA26', 'RSI', 'MACD', 'MACD_signal', 'MACD_hist', 'upper_band',
                'middle_band', 'lower_band', 'ADX', 'stoch_k', 'stoch_d']

    # Eksik değerleri temizleme
    df.dropna(inplace=True)

    X = df[features].values
    y = df['target'].values

    # Verileri normalize etme
    scaler = MinMaxScaler()
    X = scaler.fit_transform(X)

    # Verileri zaman serisi formatına dönüştürme
    X = np.array([X[i:i + 10] for i in range(len(X) - 10)])  # 10 zaman adımı
    y = y[10:]  # Y hedefleri

    return X, y


def create_lstm_model(input_shape):
    model = Sequential()
    model.add(Input(shape=input_shape))  # Input layer
    model.add(LSTM(50, return_sequences=True))
    model.add(LSTM(50))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(optimizer=Adam(), loss='binary_crossentropy', metrics=['accuracy'])
    return model


# Model eğitimi ve testi
models = {}
wallets = {}
for coin in coins:
    try:
        df = fetch_and_analyze(coin)
        if df.empty:
            print(f"No data available for {coin} for training.")
            continue

        X, y = prepare_data(df)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        model = create_lstm_model((X_train.shape[1], X_train.shape[2]))
        model.fit(X_train, y_train, epochs=10, batch_size=32, verbose=1)

        y_pred = (model.predict(X_test) > 0.5).astype(int)
        accuracy = np.mean(y_test == y_pred)
        print(f"{coin} Model Doğruluk: {accuracy:.2f}")

        models[coin] = model
    except Exception as e:
        print(f"Error training model for {coin}: {e}")

    #     wallets[coin] = 0  # Initialize with 0
    # except Exception as e:
    #     print(f"Error training model for {coin}: {e}")


def execute_trades(coin):
    try:
        df = fetch_and_analyze(coin)
        if df.empty:
            print(f"No data to execute trades for {coin}")
            return

        X, _ = prepare_data(df)
        latest_data = X[-1].reshape(1, X.shape[1], X.shape[2])
        model = models.get(coin)
        if model is None:
            print(f"No model found for {coin}")
            return

        prediction = model.predict(latest_data)[0][0]

        # Wallet balance
        balance = fetch_wallet_balance()
        wallet_coin_balance = balance.get(coin.split('/')[0], 0)

        quantity = 0.01
        if prediction > 0.5 and wallet_coin_balance > 0:
            print(f"{coin} - Alım Sinyali")
            order = exchange.create_market_buy_order(coin, quantity)
            print(f"Alım emri gönderildi: {order}")
        elif prediction <= 0.5 and wallets.get(coin, 0) > 0:
            print(f"{coin} - Satım Sinyali")
            order = exchange.create_market_sell_order(coin, quantity)
            print(f"Satım emri gönderildi: {order}")
    except Exception as e:
        print(f"Error executing trades for {coin}: {e}")


def check_for_stop_loss(coin):
    try:
        df = fetch_and_analyze(coin)
        if df.empty:
            print(f"No data to check stop-loss for {coin}")
            return

        latest_close = df['close'].iloc[-1]
        purchase_price = df['close'].iloc[-2]  # Örnek olarak son kapanış fiyatını kullanıyoruz

        # %30'dan fazla düşüş kontrolü
        if (purchase_price - latest_close) / purchase_price > 0.30:
            print(f"{coin} - Stop-Loss: Coin %30'dan fazla düştü. Satım yap!")
            # Stop-loss emri gönderme
            quantity = 0.01  # Örnek miktar, gerçek kullanımda dikkatli olun
            order = exchange.create_market_sell_order(coin, quantity)
            print(f"Stop-loss satım emri gönderildi: {order}")
    except Exception as e:
        print(f"Error checking stop-loss for {coin}: {e}")


# Sürekli Çalışacak Bot Döngüsü
while True:
    for coin in coins:
        execute_trades(coin)
        check_for_stop_loss(coin)

    # Belirli bir süre bekle (örneğin, 5 dakika = 300 saniye)
    time.sleep(300)
