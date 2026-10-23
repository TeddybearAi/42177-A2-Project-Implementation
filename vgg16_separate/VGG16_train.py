# imports the components required for model training. 
# Cross-entropy loss is used to measure classification error, 
# SGD is used to optimise model weights during learning, 
# cosine annealing is used to gradually reduce the learning rate throughout training to improve convergence.
import torch
from torch.nn import CrossEntropyLoss
from torch.optim import SGD
from torch.optim.lr_scheduler import CosineAnnealingLR

# imports the model and data loader functions.
from VGG16_model import VGG16Model
from VGG16_data import training_data_loader, testing_data_loader

# Store training and testing performance throughout the training process.
train_losses = []
train_accs   = []
test_losses  = []
test_accs    = []

# The function trains the VGG16 model on the PlantVillage dataset for a specified number of epochs and learning rate.
def train_model(epochs=30, lr=0.001):
    # Set the device to GPU if available, otherwise use CPU.
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load the training and testing data loaders, along with the class labels.
    train_loader, classes = training_data_loader()
    test_loader, _ = testing_data_loader()

    # Initialize the model with the correct number of classes.
    model = VGG16Model(n=len(classes)).to(device)

    # Define the loss function, optimizer, and learning rate scheduler.
    criterion = CrossEntropyLoss()
    optimizer = SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs)

    # Track the best accuracy achieved during training to save the best model weights.
    best_acc = 0.0

    # Loop through the specified number of epochs to train the model.
    for epoch in range(1, epochs+1):

        # Train the model for one epoch.
        model.train()
        total_loss, correct, total = 0.0, 0, 0

        # Loop through the training data loader to perform forward and backward passes 
        # update model weights, and calculate training loss and accuracy.
        for imgs, lbls in train_loader:
            imgs, lbls = imgs.to(device), lbls.to(device)

            # Zero the gradients
            # perform a forward pass
            # compute the loss
            # perform a backward pass
            # update the model weights.
            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, lbls)
            loss.backward()
            optimizer.step()

            # Accumulate the total loss and correct predictions to calculate training loss and accuracy for the epoch.
            total_loss += loss.item() * imgs.size(0)
            correct += (outputs.argmax(1) == lbls).sum().item()
            total += imgs.size(0)

        # Calculate the average training loss and accuracy for the epoch and store them in the respective lists.
        train_loss = total_loss / total
        train_acc  = correct / total

        # Store the training loss and accuracy for the epoch in the respective lists.
        train_losses.append(train_loss)
        train_accs.append(train_acc)

        # TEST
        model.eval()
        total_loss, correct, total = 0.0, 0, 0

        # Loop through the testing data loader to perform forward passes and calculate testing loss and accuracy.
        with torch.no_grad():
            for imgs, lbls in test_loader:
                imgs, lbls = imgs.to(device), lbls.to(device)
                outputs = model(imgs)
                loss = criterion(outputs, lbls)

                # Accumulate the total loss and correct predictions to calculate testing loss and accuracy for the epoch.
                total_loss += loss.item() * imgs.size(0)
                correct += (outputs.argmax(1) == lbls).sum().item()
                total += imgs.size(0)

        # Calculate the average testing loss and accuracy for the epoch and store them in the respective lists.
        test_loss = total_loss / total
        test_acc  = correct / total

        # Store the testing loss and accuracy for the epoch in the respective lists.
        test_losses.append(test_loss)
        test_accs.append(test_acc)

        # Update the learning rate scheduler to adjust the learning rate for the next epoch.
        scheduler.step()

        # Print the training and testing loss and accuracy for the epoch, and save the model weights if the testing accuracy is the best achieved so far.
        print(f"Epoch {epoch:02d} | "
              f"Train Loss {train_loss:.4f} | Train Acc {train_acc:.4f} | "
              f"Test Loss {test_loss:.4f} | Test Acc {test_acc:.4f}")

        # Save the model weights if the testing accuracy is the best achieved so far.
        if test_acc > best_acc:
            best_acc = test_acc
            torch.save(model.state_dict(), "best_model.pt")
            print("Saved best model!")

    # Return the trained model after completing all epochs.
    return model
