# PyTorch provides the framework for loading the trained model
# cross-entropy loss is used to measure prediction error during verification.
# Imports the custom VGG16 architecture and the dataset loaders required for verification. 
# Training dataset loader is used to retrieve the original class structure used during model training.
import torch
from torch.nn import CrossEntropyLoss
from VGG16_model import VGG16Model
from VGG16_data import verification_data_loader, training_data_loader

# Function to verify the trained model on the verification dataset
def verify_model():
    # Set the device to GPU if available, otherwise use CPU.
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Verification dataset (PlantDoc)
    verify_loader, _ = verification_data_loader()

    # Load TRAINING class count
    _, train_classes = training_data_loader()

    # Correct model constructor
    model = VGG16Model(n=len(train_classes)).to(device)

    # Load trained weights
    model.load_state_dict(torch.load("best_model.pt"))
    model.eval()

    # Define the loss function for verification
    criterion = CrossEntropyLoss()

    # Initialize variables to track total loss, correct predictions, and total samples for verification.
    total_loss, correct, total = 0.0, 0, 0

    # Loop through the verification data loader to perform forward passes and calculate verification loss and accuracy.
    with torch.no_grad():
        for imgs, lbls in verify_loader:
            imgs, lbls = imgs.to(device), lbls.to(device)
            outputs = model(imgs)
            loss = criterion(outputs, lbls)

            # Accumulate the total loss and correct predictions to calculate verification loss and accuracy for the dataset.
            total_loss += loss.item() * imgs.size(0)
            correct += (outputs.argmax(1) == lbls).sum().item()
            total += imgs.size(0)

    # Calculate the average verification loss and accuracy for the dataset and print the results.
    print(f"Verification Loss: {total_loss/total:.4f}")
    print(f"Verification Accuracy: {correct/total:.4f}")
