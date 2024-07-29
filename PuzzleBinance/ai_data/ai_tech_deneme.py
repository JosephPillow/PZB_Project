import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler

coins = ['BTC/USDT', 'ETH/USDT', 'BNB/USDT', 'ADA/USDT', 'XRP/USDT', 'DOGE/USDT', 'DOT/USDT', 'UNI/USDT', 'LTC/USDT', 'LINK/USDT']

def prepare_data(coin):
    table_name = coin.replace('/','_')
    df = pd.read_csv(f'C:\\Users\\zengi\\OneDrive\\Masaüstü\\PuzzleBinance\\CSV Data\\csvs\\{table_name}.csv', sep='\t')

    # Sütunları float'a dönüştür
    for col in df.columns:
        if df[col].dtype == 'object':
            try:
                df[col] = df[col].astype(float)
            except ValueError:
                print(f".")

    # NaN ve sonsuz değerleri temizle
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(inplace=True)

    features = ['SMA50', 'SMA200', 'EMA12', 'EMA26', 'RSI', 'MACD', 'MACD_signal', 'MACD_hist', 'upper_band',
                'middle_band', 'lower_band', 'ADX', 'stoch_k', 'stoch_d']
    x = df[features]
    y = df['target']

    # Verileri ölçeklendirme
    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(x)

    return x_scaled, y

# Modeli eğitme
for coin in coins:
    x, y = prepare_data(coin)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.15, random_state=31)
    model1 = LogisticRegression(max_iter=5000, solver='liblinear')  # solver='liblinear' veya başka bir solver deneyebilirsiniz
    model1.fit(x_train, y_train)
    y_pred = model1.predict(x_test)

    # Doğruluk hesaplama
    accuracy = accuracy_score(y_test, y_pred)
    print(f'{coin} model doğruluk: {accuracy}')
