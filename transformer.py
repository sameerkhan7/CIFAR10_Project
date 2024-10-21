import torch
import torchvision.datasets as datasets
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms as tfs
from transformers import ViTForImageClassification, ViTConfig
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
import random


def get_data(download_data, batch_size, image_size):
    '''
    Download CIFAR-10 Dataset into train and test splits. Create dataloaders for data.
    '''
    print("Downloading Dataset...")
    
    train_transform = tfs.Compose([
        tfs.ToTensor(),
        tfs.RandomHorizontalFlip(),
        tfs.RandomCrop(32, padding=4),
        tfs.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        tfs.Resize((image_size, image_size))
    ]) 
    test_transform = tfs.Compose([
        tfs.ToTensor(),
        tfs.Resize((image_size, image_size)),
        tfs.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ]) 
                                 
    data_train = datasets.CIFAR10(root='./data', train=True, download=download_data, transform=train_transform)
    data_test = datasets.CIFAR10(root='./data', train=False, download=download_data, transform=test_transform)
    
    train_loader = DataLoader(data_train, batch_size=batch_size,shuffle=True, num_workers=2)
    test_loader = DataLoader(data_test, batch_size=batch_size,shuffle=False, num_workers=2)
    print("Dataset Downloaded!")
    
    return train_loader, test_loader

def create_ViTModel():
    '''
    Smaller Version of ViT model with accomodations for CIFAR-10 data
    '''
    image_size = 32 # image size
    patch_size = 4 # patch size for encoding
    num_classes = 10 # num of classes
    
    #ViT model for CIFAR-10
    config = ViTConfig(
        image_size=image_size,  # Input image size
        patch_size=patch_size,  # Patch size for the 32x32 images
        num_labels=num_classes,  # Number of output classes
        hidden_size=256,  # Hidden size (smaller model for smaller images)
        num_hidden_layers=6,  # Fewer layers, since the image size is smaller
        num_attention_heads=8,  # Number of attention heads
        intermediate_size=512,  # Intermediate size of feedforward network
        hidden_dropout_prob=0.0,  # Dropout probability
        attention_probs_dropout_prob=0.0,  # Dropout on attention scores
    )
    
    model = ViTForImageClassification(config)
    
    return model

def train_model(train_data, test_data, model, optimizer, criterion, epochs, model_type):
    print("Starting Training...")
    model.train()
    batch_losses = []
    epoch_losses = []
    test_accuracies = []
    num_batches = len(train_data) # num of batches per epoch
    
    for epoch in range(epochs):
        running_loss = 0.0
        for batch_idx, (data, target) in tqdm(enumerate(train_data)):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output.logits, target)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() #running loss for epoch loss

            # Output batch loss every 100 epochs (pretrained only)
            if model_type == "pretrained":
                if (batch_idx) % 100 == 0:
                    total_batch = (batch_idx+1 + (epoch)*num_batches) # batch number over epochs
                    # print batch loss
                    print(f'Batch {batch_idx} Loss: {loss.item()}')
                    batch_losses.append([total_batch, loss.item()])
                    
                    # Get test accuracy at this step
                    accuracy, _ = test_model(test_data, model)
                    test_accuracies.append(accuracy)
                
        print(f'Epoch: {epoch + 1}, Average Training Loss: {running_loss/num_batches}' )
        # get average loss over epoch (scratch only)
        if model_type == "scratch":
            epoch_losses.append([epoch, running_loss/num_batches])
            accuracy, _ = test_model(test_data, model)
            test_accuracies.append(accuracy)
    
    print("Finished Training!")
    return model, batch_losses, epoch_losses, test_accuracies

def test_model(test_data, model):
    print("Testing Model...")
    correct = 0
    predictions = []
    with torch.no_grad():
        for batch_idx, (data, target) in tqdm(enumerate(test_data)):
            data, target = data.to(device), target.to(device)
            outputs = model(data)
            _, predicted = torch.max(outputs.logits, 1)
            predictions.append(predicted)
            correct += (predicted == target).sum().item()

    accuracy = 100 * correct / 10000
    print(f'Test Accuracy: {accuracy}%' )
    
    return accuracy, predictions

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

def train_new_model(download_data, device, model_type, seed):
    '''
    Main function to run model
    '''
    # reproducibility
    set_seed(seed)
    
    # model details
    
    if model_type == "pretrained":
        batch_size = 20
        image_size = 224
        epochs = 2   
        lr = 3e-6
        # model = ViTForImageClassification.from_pretrained("google/vit-base-patch16-224",num_labels=10, ignore_mismatched_sizes=True).to(device)
        model = ViTForImageClassification.from_pretrained("WinKawaks/vit-tiny-patch16-224",num_labels=10, ignore_mismatched_sizes=True).to(device)
    elif model_type == "scratch":
        batch_size = 100
        image_size = 32
        epochs = 40
        lr = 1e-4
        model = create_ViTModel().to(device)
    else:
        print("Not a model type!")
        return
    
    classes = ('plane', 'car', 'bird', 'cat','deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    
    # dataloaders
    train_loader, test_loader = get_data(download_data, batch_size, image_size)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    
    trained_model, batch_losses, epoch_losses, test_accuracies = train_model(train_loader, test_loader, model, optimizer, criterion, epochs, model_type)
    final_accuracy, predictions = test_model(test_loader, trained_model)
    
    # plot models and save them
    if model_type == "pretrained":
        #Plot Training Loss
        plt.figure(figsize=(10, 5))
        plt.plot(np.array(batch_losses)[:,0], np.array(batch_losses)[:,1], label='Training Loss')
        plt.title('Training Loss over Batches')
        plt.xlabel('Batch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
        plt.show()
        
        #print(np.array(batch_losses)[:,1])
        #print(np.array(test_accuracies))
        
        # Plot Test Accuracy
        plt.figure(figsize=(10, 5))
        plt.plot(np.array(batch_losses)[:,0], np.array(test_accuracies), label='Test Accuracy')
        plt.title('Test Accuracy over Batches')
        plt.xlabel('Batch')
        plt.ylabel('Accuracy(%)')
        plt.legend()
        plt.grid(True)
        plt.show()
        
        torch.save(model, f'./res/model_pretrained_{seed}.pth')
        
    if model_type == "scratch":
        #Plot Training Loss
        plt.figure(figsize=(10, 5))
        plt.plot(np.array(epoch_losses)[:,0], np.array(epoch_losses)[:,1], label='Training Loss')
        plt.title('Training Loss over Batches')
        plt.xlabel('Batch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
        plt.show()
        
        
        # Plot Test Accuracy
        plt.figure(figsize=(10, 5))
        plt.plot(np.array(epoch_losses)[:,0], np.array(test_accuracies), label='Test Accuracy')
        plt.title('Test Accuracy over Batches')
        plt.xlabel('Batch')
        plt.ylabel('Accuracy(%)')
        plt.legend()
        plt.grid(True)
        plt.show()
        
        torch.save(model, f'./res/model_scratch_{seed}.pth')
    
    # Confusion matrix
    
if __name__ == "__main__": 
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Device:", device) # cuda?
    #model_type = "scratch" # model type can be either "pretrained" or scratch
    model_type = "scratch"
    seed = 42
    train_new_model(download_data=False, device=device, model_type=model_type, seed=seed)
