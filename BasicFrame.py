import ccxt
import pandas as pd
import numpy as np
import talib as ta
import sqlite3
from sklearn.ensemble import RandomForestClassifier
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
    'apiKey': 'your_api_key',
    'secret': 'your_secret_key'
})

# İnceleme yapılacak kripto paralar
symbols = ['BTC/USDT', 'ETH/USDT', 'BNB/USDT', 'ADA/USDT', 'XRP/USDT', 'DOGE/USDT', 'DOT/USDT', 'UNI/USDT', 'LTC/USDT', 'LINK/USDT']

def fetch_and_analyze(symbol):
    bars = exchange.fetch_ohlcv(symbol, timeframe='1d', limit=365)
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
    df['ichimoku_conversion'], df['ichimoku_base'], df['ichimoku_span_a'], df['ichimoku_span_b'] = ta.ICHIMOKU(df['high'], df['low'])

    return df

# 2. Verileri SQLite Veritabanına Kaydetme
conn = sqlite3.connect('crypto_analysis.db')
c = conn.cursor()

# Tabloları oluşturma
for symbol in symbols:
    table_name = symbol.replace('/', '_')
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
            stoch_d REAL,
            ichimoku_conversion REAL,
            ichimoku_base REAL,
            ichimoku_span_a REAL,
            ichimoku_span_b REAL
        )
    """)

# Verileri çekip veritabanına kaydetme
for symbol in symbols:
    df = fetch_and_analyze(symbol)
    table_name = symbol.replace('/', '_')
    df.to_sql(table_name, conn, if_exists='replace', index_label='timestamp')

conn.commit()

# 3. Makine Öğrenmesi Modeli Geliştirme

def prepare_data(symbol):
    table_name = symbol.replace('/', '_')
    df = pd.read_sql(f"SELECT * FROM {table_name}", conn, parse_dates=['timestamp'])
    df.dropna(inplace=True)
    df['target'] = np.where(df['close'].shift(-1) > df['close'], 1, 0)  # Basit hedef değişkeni
    features = ['SMA50', 'SMA200', 'EMA12', 'EMA26', 'RSI', 'MACD', 'MACD_signal', 'MACD_hist', 'upper_band', 'middle_band', 'lower_band', 'ADX', 'stoch_k', 'stoch_d', 'ichimoku_conversion', 'ichimoku_base', 'ichimoku_span_a', 'ichimoku_span_b']
    X = df[features]
    y = df['target']
    return X, y

# Model eğitimi ve testi
models = {}
for symbol in symbols:
    X, y = prepare_data(symbol)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"{symbol} Model Accuracy: {accuracy:.2f}")
    models[symbol] = model

# 4. Modeli Güncel Verilerle Test Etme ve 5. Anlık Alım-Satım İşlemleri

def execute_trades(symbol):
    df = fetch_and_analyze(symbol)
    table_name = symbol.replace('/', '_')
    df.to_sql(table_name, conn, if_exists='replace', index_label='timestamp')
    
    # Son güncel veriyi kullanarak tahmin yapma
    latest_data = df.iloc[-1].drop(['close', 'target'])
    model = models[symbol]
    prediction = model.predict([latest_data])[0]
    
    # Alım ve satım emirleri
    quantity = 0.01  # Örnek miktar, gerçek kullanımda dikkatli olun

    if prediction == 1:
        print(f"{symbol} - Alım Sinyali")
        # Alım emri gönderme
        order = exchange.create_market_buy_order(symbol, quantity)
        print(f"Alım emri gönderildi: {order}")
    else:
        print(f"{symbol} - Satım Sinyali")
        # Satım emri gönderme
        order = exchange.create_market_sell_order(symbol, quantity)
        print(f"Satım emri gönderildi: {order}")

# 6. Kaçış Yöntemi Eklemek
def check_for_stop_loss(symbol):
    df = fetch_and_analyze(symbol)
    latest_close = df['close'].iloc[-1]
    purchase_price = df['close'].iloc[-2]  # Örnek olarak son kapanış fiyatını kullanıyoruz

    # %30'dan fazla düşüş kontrolü
    if (purchase_price - latest_close) / purchase_price > 0.30:
        print(f"{symbol} - Stop-Loss: Coin %30'dan fazla düştü. Satım yap!")
        # Stop-loss emri gönderme
        quantity = 0.01  # Örnek miktar, gerçek kullanımda dikkatli olun
        order = exchange.create_market_sell_order(symbol, quantity)
        print(f"Stop-loss satım emri gönderildi: {order}")

# Sürekli Çalışacak Bot Döngüsü
while True:
    # Tüm coinler için işlemleri gerçekleştir
    for symbol in symbols:
        execute_trades(symbol)
        check_for_stop_loss(symbol)
    
    # Belirli bir süre bekle (örneğin, 1 saat = 3600 saniye)
    time.sleep(3600)
