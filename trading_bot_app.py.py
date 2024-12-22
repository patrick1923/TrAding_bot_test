import ccxt
import pandas as pd
import numpy as np
import time
import json
import websocket
from datetime import datetime, timedelta
import logging


# Configure API keys and exchange
api_key = '6fgT4NZPtvT7ByjuqZar6vC3TpQn8RYSOtWDkKyOq17EXRfmJVZecKpWKsvpibCN'
api_secret = 'm2JLZEzndijOABGxYO0zKwjcnjIqqM0Ns2H3TVxwB29Lk2AifAlVuevBkRcqhc1f'
exchange = ccxt.binance({
    'apiKey': api_key,
    'secret': api_secret,
    'options': {'defaultType': 'future'}  # Ensure we're using Binance Futures
})

# Bot parameters
stop_loss_percentage = 0.2  # 2% stop loss
timeframe_higher = '4h'  # Changed to 4-hour timeframe
timeframe_lower = '1m'    # Changed to 1-minute timeframe
min_volume = 500_000_000  # Minimum 24-hour volume in USDT

# Configure logging
logging.basicConfig(filename='trading_bot.log', level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')

# Global variable to store the current order and market data
current_order = None
market_data = {}


def get_top_pairs(limit=45):
    # Define a list of known meme coin symbols
    meme_coins = ['DOGE', 'SHIB', 'PEPE', 'KNC'
                  'ADA', 'SAFE',  'SUI']  # Add more as needed

    # Define a list of known oracle coin symbols
    oracle_coins = ['BAND', 'BEL', 'AAVE', 'API3']  # Add more as needed

    markets = exchange.fetch_markets()
    # Filter for USDT pairs and check if they are meme or oracle coins
    relevant_pairs = [
        market for market in markets
        if '/USDT' in market['symbol'] and
        (any(coin in market['symbol'] for coin in meme_coins) or
         any(coin in market['symbol'] for coin in oracle_coins))
    ]

    # Log the number of relevant pairs found
    print(f"Found {len(relevant_pairs)} relevant pairs (meme or oracle).")

    # Sort pairs by 24h volume
    sorted_pairs = sorted(relevant_pairs, key=lambda x: float(
        x['info'].get('quoteVolume', 0)), reverse=True)

    # Log the number of pairs that meet the volume criteria
    print(f"Filtered down to {len(sorted_pairs)} relevant pairs.")

    return [market['symbol'] for market in sorted_pairs[:limit]]


def fetch_data(symbol, timeframe, limit=100):
    try:
        ohlcv = exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
        df = pd.DataFrame(
            ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        return df
    except Exception as e:
        logging.error(f"Error fetching data for {symbol}: {e}")
        return None  # Return None if there's an error

# Analyze price movement and determine bias


def analyze_price_movement(df):
    previous_close = df['close'].iloc[-2]
    current_close = df['close'].iloc[-1]
    pdh = df['high'].iloc[-2]  # Previous 4-hour high
    pdl = df['low'].iloc[-2]   # Previous 4-hour low

    if current_close > pdh:
        return "strong_bullish"
    elif current_close < pdl:
        return "strong_bearish"
    elif current_close > previous_close:
        return "bullish"
    elif current_close < previous_close:
        return "bearish"
    else:
        return "neutral"

# Identify key levels


def identify_key_levels(df):
    pdh = df['high'].iloc[-2]  # Previous 4-hour high
    pdl = df['low'].iloc[-2]   # Previous 4-hour low
    return pdh, pdl

# Function to find entry points


def find_entry(df_lower, bias, pdh, pdl):
    for i in range(1, len(df_lower)):
        if bias == "bullish":
            if df_lower['low'].iloc[i] < pdl:
                return df_lower['close'].iloc[i]  # Return only the entry price
        elif bias == "bearish":
            if df_lower['high'].iloc[i] > pdh:
                return df_lower['close'].iloc[i]
                # Return only the entry price
    return None  # Return None if no entry price is found

# Fetch minimum order size for the symbol


def get_min_order_size(symbol):
    market = exchange.market(symbol)
    return market['limits']['amount']['min']

# Place order via REST API


def place_order(symbol, side, price, quantity, stop_loss_price=None):
    global current_order
    notional_value = price * quantity
    min_order_size = get_min_order_size(symbol)

    if quantity < min_order_size:
        print(
            f"Cannot place order. Quantity {quantity} is less than minimum order size {min_order_size}.")
        return

    if notional_value < 1:
        print(
            f"Cannot place order. Notional value {notional_value:.2f} is less than 1.")
        return

    try:
        # Place the main limit order
        order = exchange.create_order(symbol, 'limit', side, quantity, price, {
            'timeInForce': 'GTC'
        })
        current_order = order
        logging.info(f"Order placed: {order}")
        print(f"Order successfully placed: {order}")

        # Place stop-loss limit order if a stop_loss_price is provided
        if stop_loss_price is not None:
            # Adjust the stop loss price to be a limit order
            stop_loss_order = exchange.create_order(symbol, 'limit', 'sell' if side == 'buy' else 'buy', quantity, stop_loss_price, {
                'timeInForce': 'GTC'
            })
            logging.info(f"Stop-loss limit order placed: {stop_loss_order}")
            print(
                f"Stop-loss limit order successfully placed: {stop_loss_order}")

    except Exception as e:
        logging.error(f"Error placing order for {symbol}: {e}")
        print(f"Error placing order for {symbol}: {e}")
# Fetch current wallet balance


def get_wallet_balance():
    balance = exchange.fetch_balance()
    return balance['USDT']['free']

# Calculate order amount as 12 % of wallet balance


def calculate_order_amount(wallet_balance, entry_price, min_order_size, max_notional_value):
    # Calculate the initial order amount as a percentage of wallet balance
    order_amount = wallet_balance * 0.8 / \
        entry_price  # Using 125% of wallet balance

    # Ensure the order amount meets the minimum order size
    if order_amount < min_order_size:
        order_amount = min_order_size

    # Calculate the notional value of the order
    notional_value = order_amount * entry_price

    # If the notional value exceeds the maximum limit, adjust the order amount
    if notional_value > max_notional_value:
        order_amount = max_notional_value / entry_price
        # Recalculate notional value after adjustment
        notional_value = order_amount * entry_price

    return order_amount, notional_value


def cancel_order(symbol):
    global current_order
    if current_order:
        try:
            exchange.cancel_order(current_order['id'], symbol)
            logging.info(f"Cancelled order: {current_order['id']}")
            current_order = None
        except Exception as e:
            logging.error(f"Error cancelling order: {e}")

# WebSocket message handler


def on_message(ws, message):
    global market_data
    data = json.loads(message)
    if 's' in data:
        symbol = data['s']
        market_data[symbol] = data

# WebSocket error handler


def on_error(ws, error):
    print(error)

# WebSocket close handler


def on_close(ws, close_status_code, close_msg):
    print("### closed ###")

# WebSocket open handler


def on_open(ws):
    subscribe_message = {
        "method": "SUBSCRIBE",
        "params": [
            f"{symbol.lower()}@ticker" for symbol in get_top_pairs()
        ],
        "id": 1
    }
    ws.send(json.dumps(subscribe_message))

# Start WebSocket connection


def start_websocket():
    global ws
    ws = websocket.WebSocketApp("wss://stream.binance.com:9443/ws",
                                on_message=on_message,
                                on_error=on_error,
                                on_close=on_close)
    ws.on_open = on_open
    ws.run_forever()


# Set display options for pandas
pd.set_option('display.max_rows', 10)  # Set maximum number of rows to display
pd.set_option('display.max_columns', None)  # Show all columns
pd.set_option('display.float_format', '{:.2f}'.format)


# Main bot logic


def trading_bot():
    print("Bot is running. Starting analysis every minute...\n")

    active_orders = {}  # Format: {symbol: {'timestamp': <time>, 'bias': <bias>}}
    ORDER_COOLDOWN = 300  # Cooldown period in seconds (5 minutes)
    MAX_NOTIONAL_VALUE = 1000
    MIN_NOTIONAL_VALUE = 1

    while True:
        try:
            current_time = datetime.utcnow() + timedelta(hours=3)
            print(f"Current UTC+3 Time: {current_time}")

            top_pairs = get_top_pairs(limit=45)
            print(f"Analyzing top futures pairs: {top_pairs}")

            wallet_balance = get_wallet_balance()
            print(f"Current Futures Wallet Balance: {wallet_balance:.2f} USDT")

            for symbol in top_pairs:
                print(f"\nAnalyzing {symbol}...")

                df_higher = fetch_data(symbol, timeframe_higher, limit=2)
                if df_higher is None:
                    print(f"Skipping {symbol} due to data fetch error.")
                    continue

                df_lower = fetch_data(symbol, timeframe_lower)
                if df_lower is None:
                    print(f"Skipping {symbol} due to data fetch error.")
                    continue

                # Analyze the current bias
                current_bias = analyze_price_movement(df_higher)
                pdh, pdl = identify_key_levels(df_higher)
                print(
                    f"4-Hour Bias: {current_bias}, Previous Day High: {pdh}, Previous Day Low: {pdl}")

                # Check if there is an active order for this symbol
                if symbol in active_orders:
                    order_info = active_orders[symbol]
                    time_since_last_order = (
                        current_time - order_info['timestamp']).total_seconds()

                    # Check if the order cooldown period has passed
                    if time_since_last_order < ORDER_COOLDOWN:
                        print(
                            f"Skipping {symbol} due to active order. Cooldown period not yet passed.")
                        continue  # Skip to the next pair

                    # Check if the bias has changed
                    if current_bias != order_info['bias']:
                        print(
                            f"Bias changed for {symbol}. Closing existing order and placing a new one.")
                        cancel_order(symbol)  # Close the existing order
                        # Remove the symbol from active orders after canceling
                        del active_orders[symbol]
                else:
                    # If there is no active order, proceed to place a new order
                    if current_bias in ["strong_bullish", "bullish"]:
                        entry_price = find_entry(
                            df_lower, current_bias, pdh, pdl)
                        if entry_price:
                            stop_loss_price = entry_price * \
                                (1 - stop_loss_percentage / 100)
                            print(
                                f"Placing new Buy Order at price {entry_price} with stop-loss at {stop_loss_price}")

                            # Fetch minimum order size for the symbol
                            min_order_size = get_min_order_size(symbol)

                            # Calculate order amount based on wallet balance, entry price, and limits
                            order_amount, notional_value = calculate_order_amount(
                                wallet_balance, entry_price, min_order_size, MAX_NOTIONAL_VALUE)

                            # Check if the notional value is below the minimum required
                            if notional_value < MIN_NOTIONAL_VALUE:
                                print(
                                    f"Cannot place order for {symbol}. Notional value {notional_value:.2f} is less than minimum required {MIN_NOTIONAL_VALUE}. Skipping to the next pair.")
                                continue

                            # Check minimum order size
                            if order_amount < min_order_size:
                                print(
                                    f"Cannot place order for {symbol}. Order amount {order_amount} is less than minimum order size {min_order_size}. Skipping to the next pair.")
                                continue

                            # Place the order with the stop-loss price
                            place_order(symbol, 'buy', entry_price,
                                        order_amount, stop_loss_price)
                            # Update active orders with the new bias and timestamp
                            active_orders[symbol] = {
                                'timestamp': current_time, 'bias': current_bias}
                        else:
                            print(
                                f"No valid entry setup for {symbol} with bullish bias.")

                    elif current_bias in ["strong_bearish", "bearish"]:
                        entry_price = find_entry(
                            df_lower, current_bias, pdh, pdl)
                        if entry_price:
                            stop_loss_price = entry_price * \
                                (1 + stop_loss_percentage / 100)
                            print(
                                f"Placing new Sell Order at price {entry_price} with stop-loss at {stop_loss_price}")

                            # Fetch minimum order size for the symbol
                            min_order_size = get_min_order_size(symbol)

                            # Calculate order amount based on wallet balance, entry price, and limits
                            order_amount, notional_value = calculate_order_amount(
                                wallet_balance, entry_price, min_order_size, MAX_NOTIONAL_VALUE)

                            # Check if the notional value is below the minimum required
                            if notional_value < MIN_NOTIONAL_VALUE:
                                print(
                                    f"Cannot place order for {symbol}. Notional value {notional_value:.2f} is less than minimum required {MIN_NOTIONAL_VALUE}. Skipping to the next pair.")
                                continue

                            # Check minimum order size
                            if order_amount < min_order_size:
                                print(
                                    f"Cannot place order for {symbol}. Order amount {order_amount} is less than minimum order size {min_order_size}. Skipping to the next pair.")
                                continue

                            # Place the order with the stop-loss price
                            place_order(symbol, 'sell', entry_price,
                                        order_amount, stop_loss_price)
                            # Update active orders with the new bias and timestamp
                            active_orders[symbol] = {
                                'timestamp': current_time, 'bias': current_bias}
                        else:
                            print(
                                f"No valid entry setup for {symbol} with bearish bias.")

            print("\nWaiting for the next analysis...")
            time.sleep(60)  # Sleep for 60 seconds before the next analysis

        except Exception as e:
            print(f"Error: {e}")
            time.sleep(60)  # Retry after 1 minute if an error occurs


if __name__ == "__main__":
    trading_bot()
