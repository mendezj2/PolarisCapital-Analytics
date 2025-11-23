"""Kafka consumer for price streams."""
import json
try:
    from kafka import KafkaConsumer
    KAFKA_AVAILABLE = True
except ImportError:
    KAFKA_AVAILABLE = False

def load_consumer_config():
    """Load consumer configuration."""
    import os
    return {
        'bootstrap_servers': os.getenv('KAFKA_BOOTSTRAP_SERVERS', 'localhost:9092').split(','),
        'group_id': 'galaxyscape-risk-consumer',
        'auto_offset_reset': 'earliest'
    }

def create_consumer(config, topic):
    """Create Kafka consumer."""
    if not KAFKA_AVAILABLE:
        return None
    
    return KafkaConsumer(
        topic,
        bootstrap_servers=config['bootstrap_servers'],
        group_id=config['group_id'],
        auto_offset_reset=config['auto_offset_reset'],
        value_deserializer=lambda m: json.loads(m.decode('utf-8'))
    )

def consume_batch(consumer, batch_size, timeout_ms=1000):
    """Consume batch of messages."""
    if not consumer:
        return []
    
    messages = []
    for message in consumer:
        messages.append(message.value)
        if len(messages) >= batch_size:
            break
    
    return messages

def process_batch(messages):
    """Process batch of messages."""
    try:
        from data_eng.streaming import stream_preprocess
        return stream_preprocess.process_price_batch(messages)
    except ImportError:
        # Fallback if import fails
        import sys
        import os
        sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
        from data_eng.streaming import stream_preprocess
        return stream_preprocess.process_price_batch(messages)

def commit_offsets(consumer):
    """Commit offsets."""
    if consumer:
        consumer.commit()

def main():
    """Main consumer loop."""
    config = load_consumer_config()
    consumer = create_consumer(config, 'prices.raw')
    
    if not consumer:
        print("Kafka not available")
        return
    
    try:
        while True:
            messages = consume_batch(consumer, batch_size=100)
            if messages:
                process_batch(messages)
                commit_offsets(consumer)
    except KeyboardInterrupt:
        consumer.close()

if __name__ == '__main__':
    main()

