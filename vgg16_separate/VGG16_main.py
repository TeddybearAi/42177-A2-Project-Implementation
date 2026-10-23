# imports the other components required for model training testing and verification.
from VGG16_train import train_model
from VGG16_test import test_model
from VGG16_verify import verify_model

# Main function to train, test, and verify the model
if __name__ == "__main__":
    model = train_model(epochs=20)
    test_model()
    verify_model()
