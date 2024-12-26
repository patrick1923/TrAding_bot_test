import ccxt
import pandas as pd
import numpy as np
import time
import json
import websocket
from datetime import datetime, timedelta
import logging
import datetime


# Configure API keys and exchange
api_key = 'your api'
api_secret = 'your api'
exchange = ccxt.binance({
    'apiKey': api_key,
    'secret': api_secret,
    'options': {'defaultType': 'future'}  # Ensure we're using Binance Futures
})

# Bot parameters
stop_loss_percentage = 20.0  # 80% stop loss
timeframe_higher = '1d'  # Changed to 1day timeframe
timeframe_lower = '1m'    # Changed to 1-minute timeframe
min_volume = 500_000_000  # Minimum 24-hour volume in USDT

# Initialize a DataFrame to log trades
trade_log = pd.DataFrame(columns=[
                         'timestamp', 'symbol', 'entry_price', 'exit_price', 'position_size', 'outcome', 'action'])


def log_trade(timestamp, symbol, entry_price, exit_price, position_size, outcome, action):
    global trade_log
    trade_log = trade_log.append({
        'timestamp': timestamp,
        'symbol': symbol,
        'entry_price': entry_price,
        'exit_price': exit_price,
        'position_size': position_size,
        'outcome': outcome,
        'action': action  # 'placed' or 'closed'
    }, ignore_index=True)


# Configure logging
logging.basicConfig(filename='trading_bot.log', level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')

# Global variable to store the current order and market data
current_order = None
market_data = {}


def get_top_pairs(limit=45):
    # Define a list of known meme coin symbols
    meme_coins = ['DOGE', 'ADA', 'PEPE', 'ETH'
                  'SUI']  # Add more as needed

    # Define a list of known oracle coin symbols
    oracle_coins = ['BTC', 'BNB', 'SOL']  # Add more as needed

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
            # Look for breakouts or retests near PDL
            if df_lower['low'].iloc[i] < pdl and df_lower['close'].iloc[i] > pdl:
                return df_lower['close'].iloc[i]  # Return only the entry price
        elif bias == "bearish":
            # Look for breakdowns or retests near PDH
            if df_lower['high'].iloc[i] > pdh and df_lower['close'].iloc[i] < pdh:
                return df_lower['close'].iloc[i]  # Return only the entry price
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
        # Assuming the order is successfully placed
        entry_price = price  # This would be the price at which the order was placed
        current_time = datetime.utcnow()

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
        log_trade(current_time, symbol, entry_price,
                  None, quantity, None, 'placed')

# Function to manually close a position


def manual_close_position(symbol, exit_price):
    global active_orders
    if symbol in active_orders:
        order_info = active_orders[symbol]
        entry_price = order_info['entry_price']
        position_size = order_info['position_size']

        # Determine if the trade was successful
        if exit_price > entry_price:  # For a buy position
            outcome = 'profit'
        elif exit_price < entry_price:  # For a sell position
            outcome = 'loss'
        else:
            outcome = 'break-even'

        # Log the closure of the trade
        log_trade(datetime.utcnow(), symbol, entry_price,
                  exit_price, position_size, outcome, 'closed')

        # Remove the order from active orders
        del active_orders[symbol]
        print(f"Position for {symbol} manually closed. Outcome: {outcome}")

# Function to automatically close a position


def auto_close_position(symbol, exit_price):
    # Similar logic to manual_close_position
    manual_close_position(symbol, exit_price)

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


# Initialize counters
skipped_pairs = 0
no_valid_entry_pairs = 0

# Initialize summary variables
total_pairs_analyzed = 0
total_skipped_pairs = 0
total_no_valid_entries = 0
total_valid_entries = 0
total_trades_executed = 0  # Track the number of trades executed
total_profit = 0.0  # Track total profit made


def trading_bot():
    global total_pairs_analyzed, total_skipped_pairs, total_no_valid_entries, total_valid_entries, total_trades_executed, total_profit

    print("Bot is running. Starting analysis every minute...\n")

    while True:
        try:
            current_time = datetime.datetime.utcnow() + timedelta(hours=3)
            print(f"Current UTC+3 Time: {current_time}")

            top_pairs = get_top_pairs(limit=45)
            # Increment total pairs analyzed
            total_pairs_analyzed += len(top_pairs)
            print(f"Analyzing top futures pairs: {top_pairs}")

            wallet_balance = get_wallet_balance()
            print(f"Current Futures Wallet Balance: {wallet_balance:.2f} USDT")

            for symbol in top_pairs:
                print(f"\nAnalyzing {symbol}...")

                # Fetch data for the 1-day timeframe
                df_higher = fetch_data(symbol, timeframe_higher, limit=2)
                if df_higher is None:
                    print(f"Skipping {symbol} due to data fetch error.")
                    total_skipped_pairs += 1  # Increment skipped pairs counter
                    continue

                # Fetch data for the 1-minute timeframe
                df_lower = fetch_data(symbol, timeframe_lower)
                if df_lower is None:
                    print(f"Skipping {symbol} due to data fetch error.")
                    total_skipped_pairs += 1  # Increment skipped pairs counter
                    continue

                # Analyze the current bias using the 1-day data
                current_bias = analyze_price_movement(df_higher)
                pdh, pdl = identify_key_levels(df_higher)
                print(
                    f"1-Day Bias: {current_bias}, Previous Day High: {pdh}, Previous Day Low: {pdl}")

                # Check for valid entry setup
                entry_price = None  # Initialize entry_price
                if current_bias in ["strong_bullish", "bullish"]:
                    entry_price = find_entry(df_lower, current_bias, pdh, pdl)
                    if entry_price:
                        print(
                            f"Valid entry setup for {symbol} (Bullish) at price: {entry_price}")

                        # Check if there is sufficient balance to place the order
                        min_order_size = get_min_order_size(symbol)
                        order_amount, notional_value = calculate_order_amount(
                            wallet_balance, entry_price, min_order_size, max_notional_value)

                        if order_amount <= 0 or wallet_balance < entry_price * order_amount:
                            print(
                                f"Insufficient balance to place order for {symbol}. Terminating the bot.")
                            generate_final_summary()  # Generate final summary report
                            exit()  # Exit the program

                        total_valid_entries += 1  # Increment valid entries counter
                        # Here you would execute the trade and update total_trades_executed and total_profit
                    else:
                        print(
                            f"No valid entry setup for {symbol} with bullish bias.")
                        total_no_valid_entries += 1  # Increment no valid entry counter
                elif current_bias in ["strong_bearish", "bearish"]:
                    entry_price = find_entry(df_lower, current_bias, pdh, pdl)
                    if entry_price:
                        print(
                            f"Valid entry setup for {symbol} (Bearish) at price: {entry_price}")

                        # Check if there is sufficient balance to place the order
                        min_order_size = get_min_order_size(symbol)
                        order_amount, notional_value = calculate_order_amount(
                            wallet_balance, entry_price, min_order_size, max_notional_value)

                        if order_amount <= 0 or wallet_balance < entry_price * order_amount:
                            print(
                                f"Insufficient balance to place order for {symbol}. Terminating the bot.")
                            generate_final_summary()  # Generate final summary report
                            exit()  # Exit the program

                        total_valid_entries += 1  # Increment valid entries counter
                        # Here you would execute the trade and update total_trades_executed and total_profit
                    else:
                        print(
                            f"No valid entry setup for {symbol} with bearish bias.")
                        total_no_valid_entries += 1  # Increment no valid entry counter

            # Summary of analysis for this cycle
            print(f"\nTotal pairs analyzed: {total_pairs_analyzed}")
            print(f"Skipped pairs: {total_skipped_pairs}")
            print(f"No valid entry setups: {total_no_valid_entries}")
            print(f"Valid entry setups: {total_valid_entries}")

            print("\nWaiting for the next analysis...")
            time.sleep(300)  # Sleep for 5 mins before the next analysis

        except Exception as e:
            print(f"Error: {e}")
            time.sleep(60)  # Retry after 1 minute if an error occurs

# Function to generate final summary when closing the bot


def generate_final_summary():
    print("\n--- Final Summary Report ---")
    print(f"Total pairs analyzed: {total_pairs_analyzed}")
    print(f"Total skipped pairs: {total_skipped_pairs}")
    print(f"Total no valid entry setups: {total_no_valid_entries}")
    print(f"Total valid entry setups: {total_valid_entries}")
    print(f"Total trades executed: {total_trades_executed}")
    print(f"Total profit made: {total_profit:.2f} USDT")
    # Assuming start_time is defined at the beginning
    print(f"Uptime: {datetime.datetime.now() - start_time}")


if __name__ == "__main__":
    start_time = datetime.datetime.now()  # Record the start time
    try:
        trading_bot()
    except KeyboardInterrupt:
        generate_final_summary()  # Call the summary function on exit
