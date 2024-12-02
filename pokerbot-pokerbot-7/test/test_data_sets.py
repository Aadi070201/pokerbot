import os
import pytest
from PIL import Image
from data_sets import normalize_image, load_data_set

@pytest.fixture
def sample_image():
    # Create a simple 32x32 image for testing purposes
    img = Image.new('L', (32, 32), color=128)
    return img

def test_normalize_image(sample_image):
    # Ensure that normalize_image properly normalizes the image to float values
    normalized_image = normalize_image(sample_image)
    
    # Assert that the normalized image has pixel values between 0 and 1
    assert normalized_image.max() <= 1.0
    assert normalized_image.min() >= 0.0

def test_load_data_set(tmpdir):
    # Prepare a temporary directory with sample image files
    data_dir = tmpdir.mkdir("training_images")
    
    # Create a simple image and save it in the directory
    img = Image.new('L', (32, 32), color=128)
    img.save(os.path.join(data_dir, "J_1.png"))
    img.save(os.path.join(data_dir, "Q_1.png"))

    # Call load_data_set() to verify it loads images and labels correctly
    training_images, training_labels, _, _ = load_data_set(data_dir, n_validation=0)
    
    assert len(training_images) == 2
    assert sorted(training_labels) == ['J', 'Q']  # Sorted to account for shuffling

def test_load_data_set_with_validation(tmpdir):
    # Prepare a temporary directory with sample image files
    data_dir = tmpdir.mkdir("training_images")

    # Create and save images for training and validation
    img = Image.new('L', (32, 32), color=128)
    img.save(os.path.join(data_dir, "J_1.png"))
    img.save(os.path.join(data_dir, "Q_1.png"))
    img.save(os.path.join(data_dir, "K_1.png"))

    # Call load_data_set() with validation examples
    training_images, training_labels, validation_images, validation_labels = load_data_set(data_dir, n_validation=1)
    
    assert len(training_images) == 2
    assert len(validation_images) == 1

    # Define the possible labels
    possible_labels = ['J', 'Q', 'K']
    
    # Assert that training labels are a subset of possible labels
    assert set(training_labels).issubset(set(possible_labels))
    
    # Assert that validation label is one of the possible labels and not in the training labels
    assert set(validation_labels).issubset(set(possible_labels))
    assert not set(validation_labels).intersection(set(training_labels))