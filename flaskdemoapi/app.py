import ccxt
from flask import Flask, render_template, request, redirect, url_for

app = Flask(__name__)

# API anahtarlarınızı buraya ekleyin
api_key = '8lrngwZQgIs9tWgC5rgxdgTNmkygxIeQN6qYAtcrHH1y36PAMWI7IvlEZUWDn3aX'
api_secret = 'KLmsyE9sa0ay5Y48SoLcIYwokJA9XOT2JtvkMxSD9KUezhBhS4MiAdcUj6Vd3dAN'

exchange = ccxt.binance({
    'apiKey': api_key,
    'secret': api_secret,
    'urls': {
        'api': {
            'public': 'https://testnet.binance.vision/api/v3/',
            'private': 'https://testnet.binance.vision/api/v3/',
        },
    },
})

exchange.set_sandbox_mode(True)

coins = ['BTC/USDT', 'ETH/USDT', 'BNB/USDT', 'ADA/USDT', 'XRP/USDT', 'DOGE/USDT', 'DOT/USDT', 'UNI/USDT', 'LTC/USDT', 'LINK/USDT']

def fetch_wallet_balance():
    try:
        balance = exchange.fetch_balance()
        return balance['total']
    except Exception as e:
        print(f"Error fetching wallet balance: {e}")
        return {}

def fetch_current_price(symbol):
    try:
        ticker = exchange.fetch_ticker(symbol)
        return ticker['last']
    except Exception as e:
        print(f"Error fetching price for {symbol}: {e}")
        return 0

@app.route('/')
def index():
    coins_data = []
    total_available_usdt = 0
    balance = fetch_wallet_balance()

    # Mevcut USDT bakiyesini al
    total_available_usdt = balance.get('USDT', 0)

    for coin in coins:
        try:
            amount = balance.get(coin.split('/')[0], 0)
            price = fetch_current_price(coin)
            usd_value = price * amount
            coins_data.append({
                'symbol': coin,
                'amount': amount,
                'usd_value': usd_value,
                'price': price
            })
        except Exception as e:
            print(f"Error processing {coin}: {e}")

    return render_template('index.html', coins_data=coins_data, total_available_usdt=total_available_usdt)

@app.route('/buy', methods=['POST'])
def buy():
    symbol = request.form['symbol']
    amount = float(request.form['amount'])
    try:
        order = exchange.create_market_buy_order(symbol, amount)
        print(f"Buy order placed: {order}")
    except Exception as e:
        print(f"Error buying {symbol}: {e}")
    return redirect(url_for('index'))

@app.route('/sell', methods=['POST'])
def sell():
    symbol = request.form['symbol']
    amount = float(request.form['amount'])
    try:
        order = exchange.create_market_sell_order(symbol, amount)
        print(f"Sell order placed: {order}")
    except Exception as e:
        print(f"Error selling {symbol}: {e}")
    return redirect(url_for('index'))

if __name__ == '__main__':
    app.run(debug=True)
