"""Streaming preprocessing."""
import math
from collections import defaultdict

_ticker_windows = defaultdict(list)
_ticker_last_price = {}

def process_price_batch(messages):
    """Process batch of price messages."""
    processed = []
    
    for msg in messages:
        ticker = msg.get('ticker')
        price = msg.get('price')
        timestamp = msg.get('timestamp')
        
        if not all([ticker, price, timestamp]):
            continue
        
        # Compute log return
        if ticker in _ticker_last_price:
            last_price = _ticker_last_price[ticker]
            log_return = compute_log_return(last_price, price)
            processed.append({
                'ticker': ticker,
                'timestamp': timestamp,
                'price': price,
                'log_return': log_return
            })
        
        _ticker_last_price[ticker] = price
        
        # Update window (keep last 60 seconds)
        _ticker_windows[ticker].append({
            'timestamp': timestamp,
            'price': price
        })
        
        # Trim window
        current_time = timestamp
        _ticker_windows[ticker] = [
            item for item in _ticker_windows[ticker]
            if current_time - item['timestamp'] < 60
        ]
    
    return processed

def compute_log_return(price_prev, price_curr):
    """Compute log return."""
    if price_prev <= 0 or price_curr <= 0:
        return 0.0
    return math.log(price_curr / price_prev)

def compute_rolling_volatility(ticker, window_seconds=60):
    """Compute rolling volatility."""
    window = _ticker_windows.get(ticker, [])
    if len(window) < 2:
        return 0.0
    
    returns = []
    for i in range(1, len(window)):
        ret = compute_log_return(window[i-1]['price'], window[i]['price'])
        returns.append(ret)
    
    if len(returns) == 0:
        return 0.0
    
    mean_return = sum(returns) / len(returns)
    variance = sum((r - mean_return) ** 2 for r in returns) / (len(returns) - 1)
    return math.sqrt(variance)

def engineer_streaming_features(processed_batch):
    """Engineer features for streaming."""
    enriched = []
    for msg in processed_batch:
        ticker = msg['ticker']
        volatility = compute_rolling_volatility(ticker)
        
        enriched.append({
            **msg,
            'volatility_60s': volatility,
            'features': {
                'log_return': msg.get('log_return', 0),
                'volatility': volatility
            }
        })
    return enriched

