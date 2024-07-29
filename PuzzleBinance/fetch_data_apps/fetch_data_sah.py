## 29.07.2024 - 16:23

import os
import ccxt
import pandas as pd
import numpy as np
import talib as ta

databinance = ccxt.binance()

# Datası çekilecek kripto paralar
coins = ['BTC/USDT', 'ETH/USDT', 'BNB/USDT', 'ADA/USDT', 'XRP/USDT', 'DOGE/USDT', 'DOT/USDT', 'UNI/USDT', 'LTC/USDT',
         'LINK/USDT']


def fetch_data(coin, timeframe='15m', target_rows=35000):
    all_bars = []
    limit = 1000  # Maksimum limit
    since = databinance.parse8601('2023-01-01T00:00:00Z')  # Başlangıç tarihi

    while len(all_bars) < target_rows:
        try:
            bars = databinance.fetch_ohlcv(coin, timeframe=timeframe, limit=limit, since=since)
            if not bars:
                break
            all_bars.extend(bars)
            if len(bars) < limit:
                break
            since = bars[-1][0] + 1
        except Exception as e:
            print(f"Hata meydana geldi: {e}")
            break

    if not all_bars:
        print(f"Veri çekilemedi: {coin}")
        return pd.DataFrame()

    df = pd.DataFrame(all_bars, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
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
    df['stoch_k'], df['stoch_d'] = ta.STOCH(df['high'], df['low'], df['close'], fastk_period=14, slowk_period=3,
                                            slowd_period=3)

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
    if not df.empty:
        sanitized_coin_name = clean_csv_name(coin)
        file_name = f"csvs/{sanitized_coin_name}.csv"
        df.to_csv(file_name, sep='\t', encoding='utf-8', index=False, header=True, index_label='timestamp')

print("Tüm CSV dosyaları 'csvs' klasörüne kaydedildi.")
