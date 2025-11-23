"""Kafka producer for price feeds."""
import json
import time
import random
try:
    from kafka import KafkaProducer
    KAFKA_AVAILABLE = True
except ImportError:
    KAFKA_AVAILABLE = False

def load_kafka_config():
    """Load Kafka configuration."""
    import os
    return {
        'bootstrap_servers': os.getenv('KAFKA_BOOTSTRAP_SERVERS', 'localhost:9092').split(','),
        'acks': 'all',
        'retries': 3
    }

def create_producer(config):
    """Create Kafka producer."""
    if not KAFKA_AVAILABLE:
        return None
    
    return KafkaProducer(
        bootstrap_servers=config['bootstrap_servers'],
        value_serializer=lambda v: json.dumps(v).encode('utf-8'),
        key_serializer=lambda k: k.encode('utf-8') if k else None,
        acks=config.get('acks', 'all'),
        retries=config.get('retries', 3)
    )

def produce_price_message(producer, ticker, price, timestamp, volume):
    """Produce price message."""
    if not producer:
        return
    
    message = {
        'ticker': ticker,
        'price': price,
        'timestamp': timestamp,
        'volume': volume
    }
    
    producer.send('prices.raw', value=message, key=ticker)
    producer.flush()

def simulate_price_feed(producer, tickers, duration_seconds=60):
    """Simulate price feed."""
    start_time = time.time()
    while time.time() - start_time < duration_seconds:
        for ticker in tickers:
            price = 100.0 + random.uniform(-5, 5)
            volume = random.randint(1000, 10000)
            produce_price_message(producer, ticker, price, time.time(), volume)
        time.sleep(1)

def main():
    """Main entrypoint."""
    config = load_kafka_config()
    producer = create_producer(config)
    
    if producer:
        tickers = ['AAPL', 'MSFT', 'GOOGL']
        simulate_price_feed(producer, tickers, duration_seconds=10)
        producer.close()
    else:
        print("Kafka not available - install kafka-python")

if __name__ == '__main__':
    main()

