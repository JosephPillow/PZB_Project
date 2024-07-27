import os
import ccxt
import pandas as pd
import numpy as np
import talib as ta

databinance = ccxt.binance()

coins = ['BTC/USDT', 'ETH/USDT', 'BNB/USDT', 'ADA/USDT', 'XRP/USDT', 'DOGE/USDT', 'DOT/USDT', 'UNI/USDT', 'LTC/USDT', 'LINK/USDT']

def fetch_data(coin):
    try:
        bars = databinance.fetch_ohlcv(coin, timeframe='1d', limit=365)
    except ValueError as ve:
        print(f"ValueError meydana geldi: {ve}")
    except TypeError as te:
        print(f"TypeError meydana geldi: {te}")
    except Exception as e:
        print(f"Başka bir hata meydana geldi: {e}")

    df = pd.DataFrame(bars, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df.set_index('timestamp', inplace=True)

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

def determine_trend(df):
    df['Signal'] = 0  # Varsayılan olarak nötr

    # RSI sinyalleri
    df.loc[df['RSI'] < 30, 'Signal'] += 1  # Al sinyali
    df.loc[df['RSI'] > 70, 'Signal'] -= 1  # Sat sinyali

    # MACD sinyalleri
    df.loc[df['MACD'] > df['MACD_signal'], 'Signal'] += 1  # Al sinyali
    df.loc[df['MACD'] < df['MACD_signal'], 'Signal'] -= 1  # Sat sinyali

    # SMA sinyalleri
    df.loc[df['SMA50'] > df['SMA200'], 'Signal'] += 1  # Al sinyali
    df.loc[df['SMA50'] < df['SMA200'], 'Signal'] -= 1  # Sat sinyali

    # EMA sinyalleri
    df.loc[df['EMA12'] > df['EMA26'], 'Signal'] += 1  # Al sinyali
    df.loc[df['EMA12'] < df['EMA26'], 'Signal'] -= 1  # Sat sinyali

    # Bollinger Bantları sinyalleri
    df.loc[df['close'] < df['lower_band'], 'Signal'] += 1  # Al sinyali
    df.loc[df['close'] > df['upper_band'], 'Signal'] -= 1  # Sat sinyali

    # ADX sinyalleri (ADX'in kendisi trendin gücünü gösterir, yön sinyali vermez)
    df.loc[df['ADX'] > 20, 'Trend_Strength'] = 1  # Güçlü trend
    df.loc[df['ADX'] <= 20, 'Trend_Strength'] = 0  # Zayıf trend

    # Stokastik Osilatör sinyalleri
    df.loc[(df['stoch_k'] < 20) & (df['stoch_d'] < 20) & (df['stoch_k'] > df['stoch_d']), 'Signal'] += 1  # Al sinyali
    df.loc[(df['stoch_k'] > 80) & (df['stoch_d'] > 80) & (df['stoch_k'] < df['stoch_d']), 'Signal'] -= 1  # Sat sinyali

    #Tüm sinyalleri fixleyerek son sinyali belirleme
    df['Final_Signal'] = df['Signal'] * df['Trend_Strength']

    return df

def evaluate_signals(df): #sinyal doğruluğu ölçme
    df['Signal_Outcome'] = np.nan

    # Al sinyalleri
    buy_signals = df[df['Final_Signal'] > 0].index
    for signal in buy_signals:
        if signal + pd.Timedelta(days=1) in df.index:
            df.loc[signal, 'Signal_Outcome'] = 1 if df.loc[signal + pd.Timedelta(days=1), 'close'] > df.loc[signal, 'close'] else 0

    # Sat sinyalleri
    sell_signals = df[df['Final_Signal'] < 0].index
    for signal in sell_signals:
        if signal + pd.Timedelta(days=1) in df.index:
            df.loc[signal, 'Signal_Outcome'] = 1 if df.loc[signal + pd.Timedelta(days=1), 'close'] < df.loc[signal, 'close'] else 0

    return df

def clean_csv_name(name):
    invalid_chars = ['\\', '/', '*', '[', ']', ':', '?']
    for char in invalid_chars:
        name = name.replace(char, '_')
    return name

os.makedirs('csvs', exist_ok=True)

for coin in coins:
    df = fetch_data(coin)
    df = determine_trend(df)
    df = evaluate_signals(df)

    # Son 5 günün sinyallerini göster
    print(f"Sinyaller for {coin}:")
    print(df[['Final_Signal', 'Signal_Outcome']].tail(5))

    sanitized_coin_name = clean_csv_name(coin)
    file_name = f"csvs/{sanitized_coin_name}.csv"
    df.to_csv(file_name, sep='\t', encoding='utf-8', index=False, header=True)

print("Tüm sinyaller ve sonuçlar hesaplandı ve CSV dosyalarına kaydedildi.")
