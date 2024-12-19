import torch
from model_13 import Model_13
from torch.utils.data import DataLoader
import torch.nn.functional as F
from train import data_set
import glob
import os

def load_latest_model():
    # Find all model files
    model_files = glob.glob('models/mnist_model_*.pth')
    if not model_files:
        print("No saved models found!")
        return None
    
    # Get the latest model file
    latest_model = max(model_files, key=os.path.getctime)
    print(f"Loading model: {latest_model}")
    
    # Load the model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Model_13().to(device)
    
    checkpoint = torch.load(latest_model)
    model.load_state_dict(checkpoint['model_state_dict'])
    test_accuracy = checkpoint['test_accuracy']
    train_accuracy = checkpoint.get('train_accuracy', 'Not saved')  # Backwards compatibility
    epoch = checkpoint['epoch']
    
    print(f"\nModel Details:")
    print(f"Epoch: {epoch}")
    print(f"Saved Training Accuracy: {train_accuracy if isinstance(train_accuracy, str) else f'{train_accuracy:.2f}%'}")
    print(f"Saved Test Accuracy: {test_accuracy:.2f}%")
    
    return model, device

def evaluate(model, device, data_loader, dataset_type="Test"):
    model.eval()
    loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in data_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss += F.nll_loss(output, target, reduction='sum').item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    loss /= len(data_loader.dataset)
    accuracy = 100. * correct / len(data_loader.dataset)

    print(f'\n{dataset_type} set: Average loss: {loss:.4f}, Accuracy: {correct}/{len(data_loader.dataset)} ({accuracy:.2f}%)\n')
    
    return accuracy

def evaluate_model():
    model, device = load_latest_model()
    if model is None:
        return
    
    # Set model to evaluation mode
    model.eval()
    
    # Load both train and test data
    train_loader, test_loader = data_set(torch.cuda.is_available())
    
    # Test the model on both datasets
    print("\nEvaluating model...")
    print("="*50)
    
    train_accuracy = evaluate(model, device, train_loader, "Training")
    test_accuracy = evaluate(model, device, test_loader, "Test")
    
    print("Final Results:")
    print("="*50)
    print(f"Training Accuracy: {train_accuracy:.2f}%")
    print(f"Test Accuracy: {test_accuracy:.2f}%")
    print("="*50)

if __name__ == "__main__":
    evaluate_model() 