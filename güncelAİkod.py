import os
import ccxt
import pandas as pd
import numpy as np
import talib as ta
import sqlite3
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import time

# Binance API bağlantısı (Testnet)
exchange = ccxt.binance({
    'options': {'defaultType': 'future'},
    'urls': {
        'api': {
            'public': 'https://testnet.binancefuture.com/fapi/v1',
            'private': 'https://testnet.binancefuture.com/fapi/v1',
        },
    },
    'apiKey': 'TgrMxFP77E1FH4RKcJ5rGXnulEq34h1pSP4UljAh4mag3Db5e73WsM13ALFDAXBt',
    'secret': 'DDwKoq5qk3cYxoaAyc5hqDk6Kore07qt0nKvun5ucEJixHzOHOsCeAQPGw6OJXB8'
})

exchange.set_sandbox_mode(True)

# İnceleme yapılacak kripto paralar
coins = ['BTC/USDT', 'ETH/USDT', 'BNB/USDT', 'ADA/USDT', 'XRP/USDT', 'DOGE/USDT', 'DOT/USDT', 'UNI/USDT', 'LTC/USDT', 'LINK/USDT']

def fetch_and_analyze(coin):
    bars = exchange.fetch_ohlcv(coin, timeframe='1d', limit=365)
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
    df['stoch_k'], df['stoch_d'] = ta.STOCH(df['high'], df['low'], df['close'], fastk_period=14, slowk_period=3, slowd_period=3)

    df['target'] = np.where(df['close'].shift(-1) > df['close'], 1, 0)

    return df

# Verileri SQLite Veritabanına Kaydetme
conn = sqlite3.connect('crypto_analysis.db')
c = conn.cursor()

# Tabloları oluşturma
for coin in coins:
    table_name = coin.replace('/', '_')
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

# Verileri çekip veritabanına kaydetme
for coin in coins:
    df = fetch_and_analyze(coin)
    table_name = coin.replace('/', '_')
    df.to_sql(table_name, conn, if_exists='replace', index_label='timestamp')

conn.commit()

def fetch_wallet_balance():
    balance = exchange.fetch_balance()
    return balance['total']

def prepare_data(coin):
    table_name = coin.replace('/', '_')
    df = pd.read_sql(f"SELECT * FROM {table_name}", conn, parse_dates=['timestamp'])

    # Object türündeki kolonları float türüne dönüştürme
    for col in df.columns:
        if df[col].dtype == 'object':
            try:
                df[col] = df[col].astype(float)
            except ValueError:
                print(f"Kolon '{col}' float'a dönüştürülemedi.")

    # NaN ve sonsuz değerleri temizleme
    df.dropna(inplace=True)  # NaN değerleri temizleme
    df.replace([np.inf, -np.inf], np.nan, inplace=True)  # Sonsuz değerleri NaN ile değiştirme
    df.dropna(inplace=True)  # NaN değerleri tekrar temizleme

    features = ['SMA50', 'SMA200', 'EMA12', 'EMA26', 'RSI', 'MACD', 'MACD_signal', 'MACD_hist', 'upper_band',
                'middle_band', 'lower_band', 'ADX', 'stoch_k', 'stoch_d']
    X = df[features]
    y = df['target']

    # NaN ve sonsuz değerleri kontrol etme (artık temizlenmiş olmalı)
    if X.isnull().values.any():
        raise ValueError("X contains NaN values")
    if np.isinf(X.values).any():
        raise ValueError("X contains infinite values")

    # Veriyi ölçekleme
    mms = MinMaxScaler()
    mms.fit(X)

    # MinMaxScaler'ın fit edilip edilmediğini kontrol etme
    if hasattr(mms, 'data_min_'):
        print("MinMaxScaler fit edildi.")
    else:
        print("MinMaxScaler fit edilmedi.")

    try:
        X_scaled = mms.transform(X)
        print("Veri başarıyla ölçeklendi.")
    except Exception as e:
        print(f"Transform sırasında hata oluştu: {e}")
        return None, None, None

    return X_scaled, y, mms

# Model eğitimi ve testi
models = {}
scalers = {}
wallets = {}
for coin in coins:
    X, y, scaler = prepare_data(coin)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = LogisticRegression(max_iter=500)  # max_iter artırıldı
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"{coin} Model Doğruluk: {accuracy:.2f}")
    models[coin] = model
    scalers[coin] = scaler

    # Wallet balance tracking
    wallets[coin] = 0  # Initialize with 0

# Modeli Güncel Verilerle Test Etme ve Anlık Alım-Satım İşlemleri
def execute_trades(coin):
    df = fetch_and_analyze(coin)
    table_name = coin.replace('/', '_')
    df.to_sql(table_name, conn, if_exists='replace', index_label='timestamp')

    # Son güncel veriyi al
    latest_data = df.iloc[-1]
    features = ['SMA50', 'SMA200', 'EMA12', 'EMA26', 'RSI', 'MACD', 'MACD_signal', 'MACD_hist',
                'upper_band', 'middle_band', 'lower_band', 'ADX', 'stoch_k', 'stoch_d']

    # `latest_data`'nın özellikleri eksiksiz olmalı 18 features hatasından kaçınmak için
    latest_data = latest_data[features].values.reshape(1, -1)

    # Veriyi ölçeklendir
    scaler = scalers[coin]
    latest_data_scaled = scaler.transform(latest_data)

    model = models[coin]
    prediction = model.predict(latest_data_scaled)[0]

    # Wallet balance
    balance = fetch_wallet_balance()
    wallet_coin_balance = balance.get(coin.split('/')[0], 0)  # Get balance for the coin

    quantity = 0.01  # Örnek miktar, gerçek kullanımda dikkatli olun
    if prediction == 1 and wallet_coin_balance > 0:
        print(f"{coin} - Alım Sinyali")
        # Alım emri gönderme
        order = exchange.create_market_buy_order(coin, quantity)
        print(f"Alım emri gönderildi: {order}")
        wallets[coin] += quantity  # Update wallet balance
    elif prediction == 0 and wallets[coin] > 0:
        print(f"{coin} - Satım Sinyali")
        # Satım emri gönderme
        order = exchange.create_market_sell_order(coin, quantity)
        print(f"Satım emri gönderildi: {order}")
        wallets[coin] -= quantity  # Update wallet balance

def check_for_stop_loss(coin):
    df = fetch_and_analyze(coin)
    latest_close = df['close'].iloc[-1]
    purchase_price = df['close'].iloc[-2]  # Örnek olarak son kapanış fiyatını kullanıyoruz

    # %30'dan fazla düşüş kontrolü
    if (purchase_price - latest_close) / purchase_price > 0.30:
        print(f"{coin} - Stop-Loss: Coin %30'dan fazla düştü. Satım yap!")
        # Stop-loss emri gönderme
        quantity = 0.01  # Örnek miktar, gerçek kullanımda dikkatli olun
        order = exchange.create_market_sell_order(coin, quantity)
        print(f"Stop-loss satım emri gönderildi: {order}")

# Sürekli Çalışacak Bot Döngüsü
while True:
    # Tüm coinler için işlemleri gerçekleştir
    for coin in coins:
        execute_trades(coin)
        check_for_stop_loss(coin)

    # Belirli bir süre bekle (örneğin, 5 dakika = 300 saniye)
    time.sleep(300)
