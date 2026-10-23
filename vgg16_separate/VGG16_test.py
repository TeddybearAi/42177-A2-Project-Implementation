# imports the components required for model training. 
# Cross-entropy loss is used to measure classification error, 
# SGD is used to optimise model weights during learning, 
# cosine annealing is used to gradually reduce the learning rate throughout training to improve convergence.
import torch
from torch.nn import CrossEntropyLoss
from VGG16_model import VGG16Model
from VGG16_data import testing_data_loader

# Function to test the trained model on the testing dataset
def test_model():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    test_loader, classes = testing_data_loader()

    # Correct model constructor
    model = VGG16Model(n=len(classes)).to(device)

    # Load trained weights
    model.load_state_dict(torch.load("best_model.pt"))
    model.eval()

    # Define the loss function for testing
    criterion = CrossEntropyLoss()

    # Initialize variables to track total loss, correct predictions, and total samples for testing.
    total_loss, correct, total = 0.0, 0, 0

    # Loop through the testing data loader to perform forward passes and calculate testing loss and accuracy.
    with torch.no_grad():
        for imgs, lbls in test_loader:
            imgs, lbls = imgs.to(device), lbls.to(device)
            outputs = model(imgs)
            loss = criterion(outputs, lbls)

            # Accumulate the total loss and correct predictions to calculate testing loss and accuracy for the dataset.
            total_loss += loss.item() * imgs.size(0)
            correct += (outputs.argmax(1) == lbls).sum().item()
            total += imgs.size(0)

    # Calculate the average testing loss and accuracy for the dataset and print the results.
    print(f"Test Loss: {total_loss/total:.4f}")
    print(f"Test Accuracy: {correct/total:.4f}")
