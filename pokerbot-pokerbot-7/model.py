import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns  # For heatmap visualization
from tensorflow import keras
from keras.models import Sequential, load_model as keras_load_model
from keras.layers import Dense, Flatten, Conv2D, MaxPooling2D
from keras.optimizers import Adam
from keras.utils import to_categorical
from keras.callbacks import ModelCheckpoint
from data_sets import * 

# Constants
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_SAVE_PATH = os.path.join(ROOT_DIR, "poker_bot_model.h5")  # Model save path with .h5 extension
NUM_CLASSES = len(LABELS)
INPUT_SHAPE = (IMAGE_SIZE, IMAGE_SIZE, 1)

def build_model():
    """
    Prepare and return a new model.
    
    Returns
    -------
    model : keras.Sequential
        The untrained model ready to be trained.
    """
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=INPUT_SHAPE),
        MaxPooling2D(pool_size=(2, 2)),

        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D(pool_size=(2, 2)),

        Flatten(),

        Dense(128, activation='relu'),
        Dense(NUM_CLASSES, activation='softmax')  # Output layer for classification
    ])    

    # Compile the model
    model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])

    return model

def plot_accuracy_heatmap(history):
    """
    Plot a heatmap for model accuracy over epochs.

    Arguments
    ---------
    history : keras.callbacks.History
        Training history object returned by model.fit().
    """
    # Extract the accuracy values
    train_acc = history.history['accuracy']
    val_acc = history.history['val_accuracy'] if 'val_accuracy' in history.history else None

    # Create data array for heatmap
    data = [train_acc]
    labels = ['Training Accuracy']
    if val_acc:
        data.append(val_acc)
        labels.append('Validation Accuracy')

    # Convert to numpy array for plotting
    data = np.array(data)

    # Create the heatmap
    plt.figure(figsize=(10, 4))
    sns.heatmap(data, annot=True, cmap='coolwarm', cbar=True, xticklabels=np.arange(1, len(train_acc) + 1),
                yticklabels=labels)
    plt.xlabel("Epoch")
    plt.title("Accuracy Heatmap per Epoch")
    plt.show()

def train_model(model, n_validation, write_to_file=False):
    """
    Train the model on the dataset.

    Arguments
    ---------
    model : keras.Sequential
        The model to be trained.
    n_validation : int
        The number of validation samples.
    write_to_file : bool
        Whether or not to save the trained model to file.

    Returns
    -------
    model : keras.Sequential
        The trained model.
    history : keras.callbacks.History
        Training history with accuracy metrics.
    """
    # Load the dataset
    training_images, training_labels, validation_images, validation_labels = \
        load_data_set(TRAINING_IMAGE_DIR, n_validation)

    # Reshape the data
    X_train = np.array(training_images).reshape(-1, IMAGE_SIZE, IMAGE_SIZE, 1)
    y_train = to_categorical([LABELS.index(label) for label in training_labels], num_classes=NUM_CLASSES)
    
    X_val = np.array(validation_images).reshape(-1, IMAGE_SIZE, IMAGE_SIZE, 1)
    y_val = to_categorical([LABELS.index(label) for label in validation_labels], num_classes=NUM_CLASSES)

    # Ensure that the directory for saving the model exists
    os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)

    # Set up a checkpoint to save the best model
    checkpoint = ModelCheckpoint(MODEL_SAVE_PATH, save_best_only=True, monitor='val_accuracy', mode='max')

    # Check if validation data is available
    validation_data = (X_val, y_val) if n_validation > 0 else None

    # Train the model
    history = model.fit(X_train, y_train, epochs=10, validation_data=validation_data, callbacks=[checkpoint])

    # Plot the accuracy heatmap
    plot_accuracy_heatmap(history)

    # Save the model if required
    if write_to_file:
        print(f"Saving model to {MODEL_SAVE_PATH}")
        model.save(MODEL_SAVE_PATH)

    return model, history

def load_model():
    """
    Load the model from the file system.

    Returns
    -------
    model : keras.Sequential or None
        The loaded model or None if no model is found.
    """
    if os.path.exists(MODEL_SAVE_PATH):
        model = keras_load_model(MODEL_SAVE_PATH)
        print("Model is loaded from", MODEL_SAVE_PATH)
        return model
    else:
        print("Model not found.")
        return None

def evaluate_model(model):
    """
    Evaluate the model on the test dataset.

    Arguments
    ---------
    model : keras.Sequential
        The trained model.

    Returns
    -------
    score : float
        The accuracy of the model on the test set.
    """
    test_images, test_labels, _, _ = load_data_set(TEST_IMAGE_DIR)
    
    X_test = np.array(test_images).reshape(-1, IMAGE_SIZE, IMAGE_SIZE, 1)
    y_test = to_categorical([LABELS.index(label) for label in test_labels], num_classes=NUM_CLASSES)

    # Evaluate the model
    score = model.evaluate(X_test, y_test, verbose=0)
    print(f"Test accuracy: {score[1]:.4f}")
    return score[1]

def identify(raw_image, model):
    """
    Identify the card rank in the given image using the trained model.

    Arguments
    ---------
    raw_image : PIL.Image
        The raw image of the card.
    model : keras.Sequential
        The trained model.

    Returns
    -------
    rank : str
        The identified rank of the card.
    """
    image = normalize_image(raw_image)
    image = np.array(image).reshape(1, IMAGE_SIZE, IMAGE_SIZE, 1)

    # Use model to predict the card rank
    prediction = model.predict(image)
    return LABELS[np.argmax(prediction)]
