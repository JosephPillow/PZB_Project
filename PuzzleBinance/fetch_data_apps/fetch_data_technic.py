import os
import ccxt
import pandas as pd
import numpy as np
import talib as ta


databinance = ccxt.binance()


# Datası çekilecek kripto paralar
coins = ['BTC/USDT', 'ETH/USDT', 'BNB/USDT', 'ADA/USDT', 'XRP/USDT', 'DOGE/USDT', 'DOT/USDT', 'UNI/USDT', 'LTC/USDT',
         'LINK/USDT']

def fetch_data(coin):
    try:
        bars = databinance.fetch_ohlcv(coin,timeframe= '1d', limit=365)
    except ValueError as ve:
        print(f"ValueError meydana geldi: {ve}")
    except TypeError as te:
        print(f"TypeError meydana geldi: {te}")
    except Exception as e:
        print(f"Başka bir hata meydana geldi: {e}")

    df = pd.DataFrame(bars, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')

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

def clean_csv_name(name):
    invalid_chars = ['\\', '/', '*', '[', ']', ':', '?']
    for char in invalid_chars:
        name = name.replace(char, '_')
    return name

os.makedirs('csvs', exist_ok=True)

for coin in coins:
    df = fetch_data(coin)
    
    sanitized_coin_name = clean_csv_name(coin)
    file_name = f"PuzzleBinance/csvs_technic/{sanitized_coin_name}.csv"

    
    df.to_csv(file_name, sep='\t', encoding='utf-8', index=False, header=True, index_label='timestamp')


print("Tüm CSV dosyaları 'csvs' klasörüne kaydedildi.")





