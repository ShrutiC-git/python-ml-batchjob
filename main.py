import os, pika, json, csv, joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression
from urllib.parse import quote_plus
from minio import Minio

RABBITMQ_HOST = os.getenv("RABBITMQ_HOST", "rabbitmq.messaging.svc.cluster.local")
RABBITMQ_USER = os.getenv("RABBITMQ_USER", "user")
RABBITMQ_PASSWORD = os.getenv("RABBITMQ_PASSWORD", "guest")
# ARTIFACT_PATH = "/artifacts/model.pkl"
DATA_PATH = "/artifacts/data.csv"


client = Minio(
    "minio.infra.svc.cluster.local:9000",
    access_key=os.getenv("MINIO_ACCESS_KEY", "minioadmin"),
    secret_key=os.getenv("MINIO_SECRET_KEY", "minioadmin123"),
    secure=False
)

BUCKET = "fraud-models"

def consume_and_train():
    events = []
    connection = None
    try:
        
        user_enc = quote_plus(RABBITMQ_USER)
        password_enc = quote_plus(RABBITMQ_PASSWORD)
        url = f"amqp://{user_enc}:{password_enc}@{RABBITMQ_HOST}:5672/"
        print(f"Connecting to: {url}")  # log for debugging

        connection = pika.BlockingConnection(pika.URLParameters(url))
        channel = connection.channel()
        channel.queue_declare(queue='checkout.events', durable=True)

        for method_frame, properties, body in channel.consume('checkout.events', inactivity_timeout=5):
            if not body:
                break
            
            event = json.loads(body)
            # We will be sending amount as a number to our query
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
    y = (df["amount"] > 2000).astype(int) # Fake threshold for fraud
    model = LogisticRegression().fit(X,y)
    
    # Ensure bucket exists
    if not client.bucket_exists(BUCKET):
        client.make_bucket(BUCKET)
    joblib.dump(model, "/tmp/model.pkl")
    client.fput_object(BUCKET, "model.pkl", "/tmp/model.pkl")
    print("Model trained and uploaded to MinIO.")
    
if __name__ == "__main__":
    consume_and_train()