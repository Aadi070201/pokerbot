import os
from model import build_model, train_model, load_model, evaluate_model
from data_sets import generate_data_set, TRAINING_IMAGE_DIR, TEST_IMAGE_DIR

def generate_datasets():
    """
    Generate training and test datasets.
    """
    print("Generating training dataset...")
    generate_data_set(500, TRAINING_IMAGE_DIR) 
    
    print("Generating test dataset...")
    generate_data_set(100, TEST_IMAGE_DIR) 

def build_and_train_model():
    """
    Build and train the model.
    """
    print("Building the model...")
    model = build_model()
    
    print("Training the model...")
    train_model(model, n_validation=100, write_to_file=True)

def load_and_evaluate_model():
    """
    Load the trained model and evaluate it.
    """
    print("Loading the trained model...")
    model = load_model()
    
    if model:
        print("Evaluating the model on test data...")
        evaluate_model(model)
    else:
        print("No trained model found. Please train the model first.")

if __name__ == "__main__":
    print("Step 1: Generate datasets")
    generate_datasets()
    
    print("Step 2: Build and train the model")
    build_and_train_model()
    
    print("Step 3: Load and evaluate the model")
    load_and_evaluate_model()
