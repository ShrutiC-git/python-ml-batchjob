import os, pika, json, csv, joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression

RABBITMQ_HOST = os.getenv("RABBITMQ_HOST", "rabbitmq.messaging.svc.cluster.local")
ARTIFACT_PATH = "/artifacts/model.pkl"
DATA_PATH = "/artifacts/data.csv"

def consume_and_train():
    events = []
    connection = None
    try:
        connection = pika.BlockingConnection(pika.ConnectionParameters(host=RABBITMQ_HOST))
        channel = connection.channel()
        channel.queue_declare(queue='checkout.events', durable=True)

        for method_frame, properties, body in channel.consume('checkout.events', inactivity_timeout=5):
            if not body:
                break
            
            event = json.loads(body)
            # Ensure amount is a number for correct processing later
            event['amount'] = float(event.get('amount', 0.0))
            events.append(event)
            channel.basic_ack(method_frame.delivery_tag) # Acknowledge message

    finally:
        if connection and not connection.is_closed:
            connection.close()
    
    if not events:
        print("No events received.")
        return
    
    with open(DATA_PATH, 'a', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=["userId", "amount"])
        if csvfile.tell() == 0:
            writer.writeheader()
        writer.writerows(events)
    
    df = pd.read_csv(DATA_PATH)
    df['amount'] = pd.to_numeric(df['amount'], errors='coerce').fillna(0)
    X = df[["amount"]]
    y = (df["amount"] > 1000).astype(int) # Fake threshold for fraud
    model = LogisticRegression().fit(X,y)
    joblib.dump(model, ARTIFACT_PATH)
    print(f"Model trained and saved to {ARTIFACT_PATH}")

if __name__ == "__main__":
    consume_and_train()