# Fraud Detection - Training Job

This `Fraud Detection` component will run as a `K8s BatchJob` in the fraud-detection workflow. The job is responsible for periodically retraining the fraud detection model. Unlike the `checkout` and `inference` services which are long-running Kubernetes Deployments, this job exits as the model finishes training.

## Workflow

The training job follows these steps:

1.  **Consume Events**: It connects to the RabbitMQ queue (`checkout.events`) provisioned by Terraform, to consume new transaction events that have occurred since the last run.
2.  **Append Data**: The new events are appended to a historical dataset (`data.csv`) stored in a persistent volume also provisioned by Terraform.
3.  **Train Model**: A new `LogisticRegression` model from `scikit-learn` is trained on the **entire** updated dataset.
4.  **Save Model**: The newly trained model artifact (`model.pkl`) is saved to a MinIO object storage bucket. The [`inference service`](https://github.com/ShrutiC-git/python-ml-inference) then loads this model for making predictions.

## Model Labeling Strategy

A crucial part of the training process is how transactions are labeled as fraudulent or not. Since we don't have pre-existing fraud labels, the script employs a simple rule-based approach for training purposes:

> **Note:** Any transaction with an `amount` greater than **1000** is automatically labeled as fraudulent (`1`), and all others are labeled as non-fraudulent (`0`).

This is implemented in the code as:
```python
y = (df["amount"] > 1000).astype(int) # Fake threshold for fraud
```