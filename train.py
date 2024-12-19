import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
import torch.nn.functional as F
from torch.utils.data import DataLoader
from model_12 import Model_12
from tqdm import tqdm
import datetime
import os
from torch.optim.lr_scheduler import StepLR
import logging
import sys

try:
    from torchsummary import summary
except ImportError:
    from torchinfo import summary

class Cutout(object):
    def __init__(self, length):
        self.length = length

    def __call__(self, img):
        h, w = img.size(1), img.size(2)
        mask = torch.ones((h, w), dtype=torch.float32)
        y = torch.randint(h, (1,))
        x = torch.randint(w, (1,))

        y1 = torch.clamp(y - self.length // 2, 0, h)
        y2 = torch.clamp(y + self.length // 2, 0, h)
        x1 = torch.clamp(x - self.length // 2, 0, w)
        x2 = torch.clamp(x + self.length // 2, 0, w)

        mask[y1:y2, x1:x2] = 0.
        mask = mask.expand_as(img)
        img = img * mask
        return img

def data_set(cuda):    
    # Train Phase transformations
    train_transforms = transforms.Compose([
                                        #  transforms.Resize((28, 28)),
                                        #  transforms.ColorJitter(brightness=0.10, contrast=0.1, saturation=0.10, hue=0.1),
                                        # transforms.RandomRotation((-8.0, 8.0), fill=(1,)),
                                        # transforms.RandomAffine(
                                        #     degrees=0, 
                                        #     translate=(0.08, 0.08),
                                        #     scale=(0.95, 1.05),
                                        #     shear=(-10, 10)
                                        # ),
                                        transforms.ToTensor(),
                                        transforms.Normalize((0.1307,), (0.3081,)),
                                        #Cutout(length=6)
                                        ])

    # Test Phase transformations
    test_transforms = transforms.Compose([
                                        #  transforms.Resize((28, 28)),
                                        #  transforms.ColorJitter(brightness=0.10, contrast=0.1, saturation=0.10, hue=0.1),
                                        transforms.ToTensor(),
                                        transforms.Normalize((0.1307,), (0.3081,))
                                        ])
    
    train = datasets.MNIST('./data', train=True, download=True, transform=train_transforms)
    test = datasets.MNIST('./data', train=False, download=True, transform=test_transforms)

    # dataloader arguments - something you'll fetch these from cmdprmt
    dataloader_args = dict(shuffle=True, batch_size=128, num_workers=4, pin_memory=True) if cuda else dict(shuffle=True, batch_size=64)
    # train dataloader
    train_loader = torch.utils.data.DataLoader(train, **dataloader_args)

    # test dataloader
    test_loader = torch.utils.data.DataLoader(test, **dataloader_args)
    return train_loader, test_loader

def model_param():
    SEED = 1
    # CUDA?
    cuda = torch.cuda.is_available()
    print("CUDA Available?", cuda)

    # For reproducibility
    torch.manual_seed(SEED)

    if cuda:
        torch.cuda.manual_seed(SEED)
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Initialize model
    print(device)
    model = Model_12().to(device)
    
    # Print model summary safely
    try:
        if device.type == "cuda":
            summary(model.cpu(), input_size=(1, 28, 28))
            model = model.to(device)  # Move back to GPU
        else:
            summary(model, input_size=(1, 28, 28))
    except Exception as e:
        print("Could not print model summary:", str(e))
        # Print basic parameter count instead
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Total parameters: {total_params}")
        print(f"Trainable parameters: {trainable_params}")
    
    return cuda, model, device

def train(model, device, train_loader, optimizer, epoch):
  model.train()
  pbar = tqdm(train_loader)
  correct = 0
  processed = 0
  for batch_idx, (data, target) in enumerate(pbar):
    # get samples
    data, target = data.to(device), target.to(device)

    # Init
    optimizer.zero_grad()
    # In PyTorch, we need to set the gradients to zero before starting to do backpropragation because PyTorch accumulates the gradients on subsequent backward passes. 
    # Because of this, when you start your training loop, ideally you should zero out the gradients so that you do the parameter update correctly.

    # Predict
    y_pred = model(data)

    # Calculate loss
    loss = F.nll_loss(y_pred, target)
    train_losses.append(loss)

    # Backpropagation
    loss.backward()
    optimizer.step()

    # Update pbar-tqdm
    
    pred = y_pred.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
    correct += pred.eq(target.view_as(pred)).sum().item()
    processed += len(data)

    pbar.set_description(desc= f'Loss={loss.item()} Batch_id={batch_idx} Train Accuracy={100*correct/processed:0.2f}')
    train_acc.append(100*correct/processed)

def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    test_losses.append(test_loss)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
    
    test_acc.append(100. * correct / len(test_loader.dataset))
    return 100. * correct / len(test_loader.dataset)

def get_timestamp():
    return datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

# Setup logging
def setup_logger():
    logging.basicConfig(
        filename=f'training_log_{get_timestamp()}.log',
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    # Also print to console
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    logging.getLogger().addHandler(console_handler)

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
    return accuracy

if __name__ == "__main__":
    setup_logger()
    logging.info("Starting training...")
    cuda, model, device = model_param()
    train_loader, test_loader = data_set(cuda)
    train_losses = []
    test_losses = []
    train_acc = []
    test_acc = []
    # optimizer = optim.Adam(model.parameters(), lr=0.01, weight_decay=1e-5)
    # scheduler = StepLR(optimizer, step_size=8, gamma=0.5)
    optimizer = optim.SGD(model.parameters(), lr=0.015, momentum=0.9)
    # scheduler = StepLR(optimizer, step_size=5, gamma=0.5)

    EPOCHS = 15
    timestamp = get_timestamp()
    model_save_path = f"models/mnist_model_{timestamp}.pth"
    best_accuracy = 0.0
    
    log_file = f"training_log_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
    with open(log_file, 'w') as f:
        sys.stdout = f
        sys.stderr = f
        for epoch in range(EPOCHS):
            logging.info(f"EPOCH: {epoch}")
            train(model, device, train_loader, optimizer, epoch)
            # scheduler.step()
            accuracy = test(model, device, test_loader)
            
            # Save model if it's the best so far
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                # Create models directory if it doesn't exist
                os.makedirs("models", exist_ok=True)
                model_save_path = f"models/mnist_model_{timestamp}.pth"
                
                # Get current training accuracy
                model.eval()
                train_accuracy = evaluate(model, device, train_loader, "Training")
                
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'test_accuracy': accuracy,
                    'train_accuracy': train_accuracy,
                    'timestamp': timestamp
                }, model_save_path)
                
                # Make the message more visible
                save_message = f"\n{'='*50}\nNew best model saved to: {model_save_path}\nEpoch: {epoch}\nTraining Accuracy: {train_accuracy:.2f}%\nTest Accuracy: {accuracy:.2f}%\n{'='*50}\n"
                print(save_message)
                logging.info(save_message)
            logging.info(f"Epoch {epoch} completed. Test Accuracy: {accuracy:.2f}%")
