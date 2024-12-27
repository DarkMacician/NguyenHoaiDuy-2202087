import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from model import Autoencoder
from model import DataPreprocessor

def precision_at_k(actual, predicted, k):
    """Calculate Precision@K."""
    pred_top_k = torch.topk(predicted, k, dim=1).indices
    matches = torch.gather(actual, 1, pred_top_k)
    precision = matches.sum(dim=1) / k
    return precision.mean().item()


def recall_at_k(actual, predicted, k):
    """Calculate Recall@K."""
    pred_top_k = torch.topk(predicted, k, dim=1).indices  # Top K predicted items
    actual_top_k = torch.topk(actual, k, dim=1).indices  # Top K actual items

    # Initialize a tensor to keep track of the matches
    matches = torch.zeros_like(predicted, dtype=torch.uint8)

    # Count matches between predicted and actual top K
    for i in range(len(pred_top_k)):
        matches[i, pred_top_k[i]] = 1  # Mark the top K predicted items as 1

    # Calculate recall: the sum of matches divided by the number of actual relevant items
    recall = torch.sum(matches * actual).item() / torch.sum(actual).item()

    return recall


def ndcg_k(actual, predicted, k):
    """Calculate NDCG@K."""
    pred_top_k = torch.topk(predicted, k, dim=1).values
    actual_top_k = torch.topk(actual, k, dim=1).values

    dcg = ((torch.pow(2, pred_top_k) - 1) / torch.log2(torch.arange(2, k + 2, dtype=torch.float32).to(predicted.device))).sum(dim=1)
    idcg = ((torch.pow(2, actual_top_k) - 1) / torch.log2(torch.arange(2, k + 2, dtype=torch.float32).to(predicted.device))).sum(dim=1)
    ndcg = (dcg / idcg.clamp(min=1e-8)).mean()  # Avoid division by zero
    return ndcg.item()

def evaluate_model(model, dataloader, criterion, k=5):
    """Evaluate the model using recommendation metrics."""
    model.eval()
    total_loss = 0.0
    all_inputs = []
    all_outputs = []

    with torch.no_grad():
        for data in dataloader:
            inputs = data[0]
            outputs = model(inputs)
            loss = criterion(outputs, inputs)
            total_loss += loss.item()
            all_inputs.append(inputs)
            all_outputs.append(outputs)

    # Combine all batches
    inputs = torch.cat(all_inputs, dim=0)
    outputs = torch.cat(all_outputs, dim=0)

    # Compute metrics
    mse_loss = total_loss / len(dataloader)
    precision = precision_at_k(inputs, outputs, k)
    recall = recall_at_k(inputs, outputs, k)
    ndcg = ndcg_k(inputs, outputs, k)

    print(f"Evaluation Metrics: Loss={mse_loss:.4f}, Precision@{k}={precision:.4f}, Recall@{k}={recall:.4f}, NDCG@{k}={ndcg:.4f}")
    return mse_loss, precision, recall, ndcg

def train_model():
    # Instantiate the data preprocessor and load the dataset
    preprocessor = DataPreprocessor(file_path='user_investment_data_v2.csv')
    preprocessor.load_data()
    preprocessor.encode_categories()
    user_features_tensor = preprocessor.preprocess_features()

    # Set input dimension (user-item interaction + spended_time + amount)
    input_dim = user_features_tensor.shape[1]
    hidden_dim = 50

    # Initialize the model
    model = Autoencoder(input_dim=input_dim, hidden_dim=hidden_dim)

    # Training settings
    epochs = 1000
    learning_rate = 0.00001
    batch_size = 30
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # DataLoader for batch processing
    dataset = torch.utils.data.TensorDataset(user_features_tensor)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Variables to track the best model
    best_precision = 0.0
    best_model_path = "best_model.pth"

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for data in dataloader:
            inputs = data[0]
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, inputs)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        avg_loss = running_loss / len(dataloader)
        if epoch % 10 == 0:
            print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")

        # Evaluate every 50 epochs
        if epoch % 50 == 0 or epoch == epochs - 1:
            mse_loss, precision, recall, ndcg = evaluate_model(model, dataloader, criterion, k=5)
            if precision > best_precision:  # Save model if Precision@K improves
                best_precision = precision
                torch.save(model.state_dict(), best_model_path)
                print(f"Epoch {epoch+1}: Best model saved with Precision@5={best_precision:.4f}")

    print(f"Training complete. Best model saved at {best_model_path} with Precision@5={best_precision:.4f}.")


if __name__ == "__main__":
    train_model()
