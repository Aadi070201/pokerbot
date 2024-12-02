import os
import pytest
import numpy as np
from tensorflow import keras
from model import build_model, load_model, identify
from data_sets import normalize_image
from PIL import Image

@pytest.fixture
def sample_image():
    # Create a simple 32x32 image for testing purposes
    img = Image.new('L', (32, 32), color=128)
    return img

def test_build_model():
    # Ensure the model builds correctly
    model = build_model()

    # Check if it's a Keras Sequential model and has layers
    assert isinstance(model, keras.Sequential)
    assert len(model.layers) > 0
    assert model.input_shape == (None, 32, 32, 1)  # Input shape matches the dataset IMAGE_SIZE

def test_load_model(tmpdir):
    # Assume a pre-trained model exists at the save path
    model_save_path = os.path.join(tmpdir, "poker_bot_model.h5")
    
    # Create and save a simple model
    model = build_model()
    model.save(model_save_path)
    
    # Test that load_model successfully loads the saved model
    loaded_model = load_model()
    assert loaded_model is not None
    assert isinstance(loaded_model, keras.Model)  # Check if it loads as a Keras model

def test_identify(sample_image, tmpdir):
    # Build and save the model for testing
    model_save_path = os.path.join(tmpdir, "poker_bot_model.h5")
    model = build_model()
    model.save(model_save_path)
    
    # Load the model for identification
    loaded_model = load_model()

    # Normalize the sample image
    normalized_image = normalize_image(sample_image)
    
    # Test that identify() returns a valid card label (J, Q, K) based on the model's prediction
    card = identify(normalized_image, loaded_model)
    assert card in ['J', 'Q', 'K']  # Assuming the model returns one of these labels

