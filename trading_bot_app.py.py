import ccxt
import pandas as pd
import numpy as np
import time
import json
import websocket
from datetime import datetime, timedelta
import logging
import datetime
import time
import threading

# Configure API keys and exchange
api_key = 'API'
api_secret = 'API'
exchange = ccxt.binance({
    'apiKey': api_key,
    'secret': api_secret,
    'options': {'defaultType': 'future'}  # Ensure we're using Binance Futures
})

stop_loss_percentage = 20.0
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


def get_top_pairs(limit=65):
    favorite_pairs = [
        'AGLD/USDT', 'ATOM/USDT', 'SXP/USDT', 'USUAL/USDT',
        'STG/USDT', 'LINK/USDT', 'ALGO/USDT', 'POWR/USDT',
        'MANA/USDT', 'SCRT/USDT', 'HIVE/USDT', 'FIL/USDT',
        'SYN/USDT', 'HBAR/USDT', 'WIF/USDT', 'ATA/USDT',
        'ZRX/USDT', 'DOT/USDT', 'SOL/USDT', 'UNI/USDT',
        'VET/USDT', 'FIL/USDT'
    ]

    markets = exchange.fetch_markets()
    relevant_pairs = [
        market['symbol'] for market in markets
        if market['symbol'] in favorite_pairs
    ]

    print(f"Found {len(relevant_pairs)} relevant pairs from favorites.")
    return relevant_pairs[:limit]


def fetch_data(symbol, timeframe, limit=100):
    try:
        ohlcv = exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
        df = pd.DataFrame(
            ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        # Calculate 14-period moving average for volume
        df['MA_14'] = df['volume'].rolling(window=14).mean()
        return df
    except Exception as e:
        logging.error(f"Error fetching data for {symbol}: {e}")
        return None

# Identify Demand and Supply Zones


def identify_zones(df):
    # Ensure we have at least two days of data
    if len(df) < 2:
        return [], []

    # Previous day's high and low
    previous_day_high = df['high'].iloc[-2]  # High of the previous day
    previous_day_low = df['low'].iloc[-2]    # Low of the previous day

    # Set demand and supply zones
    # Demand zone is the previous day's high
    demand_zones = [previous_day_high]
    # Supply zone is the previous day's low
    supply_zones = [previous_day_low]

    return demand_zones, supply_zones


def is_hammer(candle):
    """Check if the given candle is a hammer."""
    body = abs(candle['close'] - candle['open'])
    lower_shadow = candle['low'] - min(candle['close'], candle['open'])
    upper_shadow = candle['high'] - max(candle['close'], candle['open'])

    # Hammer criteria
    return (lower_shadow >= 2 * body) and (upper_shadow <= body)


def is_bullish_engulfing(candle1, candle2):
    """Check if the two given candles form a bullish engulfing pattern."""
    return (candle1['close'] < candle1['open'] and  # First candle is bearish
            candle2['close'] > candle2['open'] and  # Second candle is bullish
            # Second candle opens below the first candle's close
            candle2['open'] < candle1['close'] and
            # Second candle closes above the first candle's open
            candle2['close'] > candle1['open'])


def is_shooting_star(candle):
    """Check if the given candle is a shooting star."""
    body = abs(candle['close'] - candle['open'])
    lower_shadow = candle['low'] - min(candle['close'], candle['open'])
    upper_shadow = candle['high'] - max(candle['close'], candle['open'])

    # Shooting star criteria
    return (upper_shadow >= 2 * body) and (lower_shadow <= body)


def is_bearish_engulfing(candle1, candle2):
    """Check if the two given candles form a bearish engulfing pattern."""
    return (candle1['close'] > candle1['open'] and  # First candle is bullish
            candle2['close'] < candle2['open'] and  # Second candle is bearish
            # Second candle opens above the first candle's close
            candle2['open'] > candle1['close'] and
            # Second candle closes below the first candle's open
            candle2['close'] < candle1['open'])


def find_entry_with_confirmation(df):
    """Find entry price with confirmation based on patterns."""
    for i in range(1, len(df)):
        # Check for Hammer
        if is_hammer(df.iloc[i]):
            return df['close'].iloc[i]  # Return the price of the hammer candle

        # Check for Bullish Engulfing
        if i > 0 and is_bullish_engulfing(df.iloc[i-1], df.iloc[i]):
            # Return the price of the engulfing candle
            return df['close'].iloc[i]

        # Check for Shooting Star
        if is_shooting_star(df.iloc[i]):
            # Return the price of the shooting star candle
            return df['close'].iloc[i]

        # Check for Bearish Engulfing
        if i > 0 and is_bearish_engulfing(df.iloc[i-1], df.iloc[i]):
            # Return the price of the engulfing candle
            return df['close'].iloc[i]

    return None
# Confirm Entry Conditions


# Find entry price with confirmation
def find_entry_with_confirmation(df_lower, demand_zones, supply_zones):
    for i in range(1, len(df_lower)):
        # Check for Long Trade
        if df_lower['close'].iloc[i] in demand_zones:
            if df_lower['close'].iloc[i] > df_lower['open'].iloc[i-1] and df_lower['open'].iloc[i-1] > df_lower['close'].iloc[i-1]:
                return df_lower['close'].iloc[i]

        # Check for Short Trade
        if df_lower['close'].iloc[i] in supply_zones:
            if df_lower['close'].iloc[i] < df_lower['open'].iloc[i-1] and df_lower['open'].iloc[i-1] < df_lower['close'].iloc[i-1]:
                return df_lower['close'].iloc[i]
    return None

    # Proceed to place the order if all conditions are met

# Check for Retest Confirmation


def check_retest_confirmation(df_lower, breakout_price, is_long):
    for i in range(len(df_lower)):
        if is_long:
            if df_lower['low'].iloc[i] <= breakout_price <= df_lower['high'].iloc[i]:
                if df_lower['close'].iloc[i] > df_lower['open'].iloc[i]:  # Bullish confirmation
                    return True
        else:
            if df_lower['low'].iloc[i] <= breakout_price <= df_lower['high'].iloc[i]:
                if df_lower['close'].iloc[i] < df_lower['open'].iloc[i]:  # Bearish confirmation
                    return True
    return False

# Check Volume Confluence


def is_volume_confirmation(df, volume_threshold=1.0):
    """
    Check if the current volume is above the 14-period MA.
    """
    current_volume = df['volume'].iloc[-1]
    ma_volume = df['MA_14'].iloc[-1]
    return current_volume > ma_volume * volume_threshold


def is_bullish_confirmation(df, demand_zones):
    """
    Check for bullish confirmation: price near demand zone + increasing volume.
    """
    current_price = df['close'].iloc[-1]
    return (current_price >= demand_zones[0]) and is_volume_confirmation(df)


def is_bearish_confirmation(df, supply_zones):
    """
    Check for bearish confirmation: price near supply zone + increasing volume.
    """
    current_price = df['close'].iloc[-1]
    return (current_price <= supply_zones[0]) and is_volume_confirmation(df)


def is_breakout_confirmation(df, breakout_price):
    """
    Check for breakout confirmation: price breaks resistance/support with high volume.
    """
    current_price = df['close'].iloc[-1]
    return (current_price > breakout_price) and is_volume_confirmation(df)

# Fetch minimum order size for the symbol


def get_min_order_size(symbol):
    market = exchange.market(symbol)
    return market['limits']['amount']['min']

# Place order via REST API


def place_order(symbol, side, price, quantity, stop_loss_price=None):
    try:
        if quantity <= 0:
            raise ValueError("Quantity must be greater than zero.")

        order = exchange.create_order(
            symbol, 'limit', side, quantity, price, {'timeInForce': 'GTC'}
        )
        logging.info(
            f"Main order placed: {side} {quantity} of {symbol} at {price}. Order ID: {order['id']}")

        # Update current_orders with the new order
        current_orders[symbol] = order['id']

        if stop_loss_price is not None:
            stop_loss_side = 'sell' if side == 'buy' else 'buy'
            stop_loss_order = exchange.create_order(
                symbol, 'limit', stop_loss_side, quantity, stop_loss_price, {
                    'timeInForce': 'GTC'}
            )
            logging.info(
                f"Stop-loss order placed: {stop_loss_side} {quantity} of {symbol} at {stop_loss_price}. Order ID: {stop_loss_order['id']}")

        return order

    except ccxt.InsufficientFunds as e:
        logging.error(
            f"Insufficient funds for placing order for {symbol}: {e}")
    except ccxt.InvalidOrder as e:
        logging.error(f"Invalid order parameters for {symbol}: {e}")
    except Exception as e:
        logging.error(f"Error placing order for {symbol}: {e}")

# Fetch current wallet balance


def get_wallet_balance():
    balance = exchange.fetch_balance()
    return balance['USDT']['free']

# Calculate order amount as 80% of wallet balance


def calculate_order_amount(wallet_balance, entry_price, min_order_size):
    # Ensure you are calling wallet_balance correctly
    order_amount = wallet_balance * 0.8 / entry_price
    if order_amount < min_order_size:
        order_amount = min_order_size
    return order_amount, order_amount * entry_price

# Cancel order


def cancel_order(symbol):
    global current_order, current_orders
    if current_order:
        try:
            exchange.cancel_order(current_order['id'], symbol)
            logging.info(f"Cancelled order: {current_order['id']}")
            current_order = None
            # Remove the order from current_orders
            if symbol in current_orders:
                del current_orders[symbol]
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
    ws = websocket.WebSocketApp("wss ://stream.binance.com:9443/ws",
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


# Global variables to track positions and orders
current_positions = {}  # {symbol: 'long' or 'short'}
current_orders = {}     # {symbol: order_id}


def trading_bot():
    global total_pairs_analyzed, total_skipped_pairs, total_no_valid_entries, total_valid_entries, total_trades_executed, total_profit, current_positions, current_orders

    print("Bot is running. Starting analysis continuously...\n")
    wallet_balance = get_wallet_balance()
    start_time = time.time()

    # Get the top pairs once at the start
    top_pairs = get_top_pairs(limit=22)

    while True:
        elapsed_time = time.time() - start_time
        elapsed_minutes = elapsed_time // 60
        elapsed_seconds = elapsed_time % 60
        print(
            f"Elapsed Time: {int(elapsed_minutes)} minutes and {int(elapsed_seconds)} seconds")

        for symbol in top_pairs:
            print(f"\nAnalyzing {symbol}...")

            # Check if there is an existing order for the symbol
            if symbol in current_orders:
                # Fetch the current order details
                try:
                    order = exchange.fetch_order(
                        current_orders[symbol], symbol)
                    if order['status'] == 'open':
                        # Re-evaluate entry conditions
                        df_1m = fetch_data(symbol, '1m', limit=14)
                        if df_1m is None:
                            print(
                                f"Skipping {symbol} due to data fetch error.")
                            continue

                        demand_zones, supply_zones = identify_zones(df_1m)
                        if is_bullish_confirmation(df_1m, demand_zones):
                            new_entry_price = df_1m['close'].iloc[-1]
                            new_stop_loss_price = new_entry_price * \
                                (1 - stop_loss_percentage / 100)
                            if new_entry_price != order['price']:
                                # Cancel the existing order
                                cancel_order(symbol)
                                # Place a new order with updated conditions
                                min_order_size = get_min_order_size(symbol)
                                order_amount, _ = calculate_order_amount(
                                    wallet_balance, new_entry_price, min_order_size)
                                place_order(symbol, 'buy', new_entry_price,
                                            order_amount, new_stop_loss_price)
                        elif is_bearish_confirmation(df_1m, supply_zones):
                            new_entry_price = df_1m['close'].iloc[-1]
                            new_stop_loss_price = new_entry_price * \
                                (1 + stop_loss_percentage / 100)
                            if new_entry_price != order['price']:
                                # Cancel the existing order
                                cancel_order(symbol)
                                # Place a new order with updated conditions
                                min_order_size = get_min_order_size(symbol)
                                order_amount, _ = calculate_order_amount(
                                    wallet_balance, new_entry_price, min_order_size)
                                place_order(symbol, 'sell', new_entry_price,
                                            order_amount, new_stop_loss_price)
                    else:
                        # Remove the order from current_orders if it's no longer open
                        del current_orders[symbol]
                except Exception as e:
                    print(
                        f"Error fetching or processing order for {symbol}: {e}")
                    continue

            # Skip if there is an open position for this symbol
            if symbol in current_positions:
                print(f"Skipping {symbol} due to existing position.")
                continue

            # Fetch and display the current price
            ticker = exchange.fetch_ticker(symbol)
            current_price = ticker['last']
            print(f"Current Price for {symbol}: {current_price:.4f}")

            # Fetch data for the daily timeframe
            df_daily = fetch_data(symbol, '1d', limit=2)
            if df_daily is None:
                print(f"Skipping {symbol} due to data fetch error.")
                total_skipped_pairs += 1
                continue

            # Identify demand and supply zones
            demand_zones, supply_zones = identify_zones(df_daily)
            print(f"Demand Zone for {symbol}: {demand_zones}")
            print(f"Supply Zone for {symbol}: {supply_zones}")

            # Fetch data for the 1-minute timeframe
            df_1m = fetch_data(symbol, '1m', limit=14)
            if df_1m is None:
                print(f"Skipping {symbol} due to data fetch error.")
                total_skipped_pairs += 1
                continue

            # Display current volume and 14-period MA
            current_volume = df_1m['volume'].iloc[-1]
            ma_volume = df_1m['MA_14'].iloc[-1]
            print(f"Current Volume for {symbol}: {current_volume:.2f}")
            print(f"14-Period Volume MA for {symbol}: {ma_volume:.2f}")

            # Check for bullish confirmation
            if is_bullish_confirmation(df_1m, demand_zones):
                entry_price = df_1m['close'].iloc[-1]
                stop_loss_price = entry_price * \
                    (1 - stop_loss_percentage / 100)
                print(
                    f"Valid bullish entry setup for {symbol} at price: {entry_price} with increasing volume.")
                side = 'buy'

            # Check for bearish confirmation
            elif is_bearish_confirmation(df_1m, supply_zones):
                entry_price = df_1m['close'].iloc[-1]
                stop_loss_price = entry_price * \
                    (1 + stop_loss_percentage / 100)
                print(
                    f"Valid bearish entry setup for {symbol} at price: {entry_price} with increasing volume.")
                side = 'sell'

            else:
                print(
                    f"Conditions not met for a valid entry for {symbol}. Skipping...")
                continue

            # Proceed to place the order
            min_order_size = get_min_order_size(symbol)
            order_amount, notional_value = calculate_order_amount(
                wallet_balance, entry_price, min_order_size)

            if order_amount < min_order_size:
                print(
                    f"Order amount {order_amount} is less than minimum order size {min_order_size}. Adjusting to minimum.")
                order_amount = min_order_size

            if order_amount <= 0 or get_wallet_balance() < entry_price * order_amount:
                print(f"Insufficient balance to place order for {symbol}.")
                total_no_valid_entries += 1
                continue

            try:
                print(
                    f"Attempting to place order: {side} {order_amount} of {symbol} at {entry_price} with stop loss at {stop_loss_price}")
                place_order(symbol, side, entry_price,
                            order_amount, stop_loss_price)
                current_positions[symbol] = 'long' if side == 'buy' else 'short'
                total_trades_executed += 1
                total_valid_entries += 1
            except Exception as e:
                print(f"Error placing order for {symbol}: {e}")

        # Summary of analysis for this cycle
        print(f"\nTotal pairs analyzed: {total_pairs_analyzed}")
        print(f"Skipped pairs: {total_skipped_pairs}")
        print(f"No valid entry setups: {total_no_valid_entries}")
        print(f"Valid entry setups: {total_valid_entries}")


def generate_final_summary():
    print("\n--- Final Summary Report ---")
    print(f"Total pairs analyzed: {total_pairs_analyzed}")
    print(f"Total skipped pairs: {total_skipped_pairs}")
    print(f"Total no valid entry setups: {total_no_valid_entries}")
    print(f"Total valid entry setups: {total_valid_entries}")
    print(f"Total trades executed: {total_trades_executed}")
    print(f"Total profit made: {total_profit:.2f} USDT")
    print(f"Uptime: {datetime.datetime.now() - start_time}")


if __name__ == "__main__":
    start_time = datetime.datetime.now()  # Record the start time
    try:
        trading_bot()
    except KeyboardInterrupt:
        generate_final_summary()  # Call the summary function on exit
