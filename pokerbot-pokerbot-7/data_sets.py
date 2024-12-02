import os
import random
from matplotlib import font_manager
from PIL import Image, ImageDraw, ImageFont
import numpy as np

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))  # Current file marks the root directory
TRAINING_IMAGE_DIR = os.path.join(ROOT_DIR, "data_sets", "training_images")  # Directory for storing training images
TEST_IMAGE_DIR = os.path.join(ROOT_DIR, "data_sets", "test_images")  # Directory for storing test images
LABELS = ['J', 'Q', 'K', 'A']  # Possible card labels
IMAGE_SIZE = 32 
ROTATE_MAX_ANGLE = 15

FONTS = [
    font_manager.findfont(font_manager.FontProperties(family = 'sans-serif', style = 'normal', weight = 'normal')),
    font_manager.findfont(font_manager.FontProperties(family = 'sans-serif', style = 'italic', weight = 'normal')),
    font_manager.findfont(font_manager.FontProperties(family = 'sans-serif', style = 'normal', weight = 'medium')),
    font_manager.findfont(font_manager.FontProperties(family = 'serif', style = 'normal', weight = 'normal')),
    font_manager.findfont(font_manager.FontProperties(family = 'serif', style = 'italic', weight = 'normal')),
    font_manager.findfont(font_manager.FontProperties(family = 'serif', style = 'normal', weight = 'medium')),
]  # True type system fonts


def normalize_image(raw_image: Image):
    """
    Normalize a raw image to serve as input to the image classifier.

    Arguments
    ---------
    raw_image : Image
        Raw image to normalize.

    Returns
    -------
    image : list/matrix/structure of floats between zero and one
        Normalized image that can be used by the image classifier.
    """

    # Convert the image to a numpy array and normalize its pixel values to be between 0 and 1
    image = np.asarray(raw_image).astype(np.float32) / 255.0
    return image


def load_data_set(data_dir, n_validation = 0):
    """
    Normalize the images in data_dir and divide in a training and validation set.

    Parameters
    ----------
    data_dir : str
        Directory of images to load
    n_validation : int
        Number of images that are assigned to the validation set
    """

    # Extract png files
    files = os.listdir(data_dir)
    png_files = []
    for file in files:
        if file.split('.')[-1] == "png":
            png_files.append(file)

    random.shuffle(png_files)  # Shuffled list of the png-file names that are stored in data_dir

    # Load the training and validation set and prepare the images and labels. Use normalize_image()
    # to normalize raw images (you can load an image with Image.open()) to be processed by your
    # image classifier. You can extract the original label from the image filename.
    training_images = []
    training_labels = []
    validation_images = []
    validation_labels = []

    for i, file in enumerate(png_files):
        img = Image.open(os.path.join(data_dir, file))  
        label = file.split('_')[0]  
        
        normalized_img = normalize_image(img)  
        
       # splitting into training and validation sets
        if i < n_validation:
            validation_images.append(normalized_img)
            validation_labels.append(label)
        else:
            training_images.append(normalized_img)
            training_labels.append(label)
    
    return training_images, training_labels, validation_images, validation_labels


def generate_data_set(n_samples, data_dir):
    """
    Generate n_samples noisy images by using generate_noisy_image(), and store them in data_dir.

    Arguments
    ---------
    n_samples : int
        Number of train/test examples to generate
    data_dir : str in [TRAINING_IMAGE_DIR, TEST_IMAGE_DIR]
        Directory for storing images
    """

    if not os.path.exists(data_dir):
        os.makedirs(data_dir)  # Generate a directory for data set storage, if not already present

    for i in range(n_samples):
        # Setting a random rank and converting it to a noisy image through generate_noisy_image().
        rank = random.choice(LABELS)
        noise_level = random.uniform(0, 1)  
        img = generate_noisy_image(rank, noise_level)
        
     
        img.save(os.path.join(data_dir, f"{rank}_{i}.png"))  



def generate_noisy_image(rank, noise_level):
    """
    Generate a noisy image with a given noise corruption. This implementation mirrors how the server generates the
    raw images. However the exact server settings for noise_level and ROTATE_MAX_ANGLE are unknown.
    For the PokerBot assignment you won't need to update this function, but remember to test it.

    Arguments
    ---------
    rank : str in ['J', 'Q', 'K']
        Original card rank.
    noise_level : float between zero and one
        Probability with which a given pixel is randomized.

    Returns
    -------
    noisy_img : Image
        A noisy image representation of the card rank.
    """

    if not 0 <= noise_level <= 1:
        raise ValueError(f"Invalid noise level: {noise_level}, value must be between zero and one")
    if rank not in LABELS:
        raise ValueError(f"Invalid card rank: {rank}")

    # Create rank image from text
    font = ImageFont.truetype(random.choice(FONTS), size = IMAGE_SIZE - 6)  # Pick a random font
    img = Image.new('L', (IMAGE_SIZE, IMAGE_SIZE), color = 255)
    draw = ImageDraw.Draw(img)
    (text_width, text_height) = draw.textsize(rank, font = font)  # Extract text size
    draw.text(((IMAGE_SIZE - text_width) / 2, (IMAGE_SIZE - text_height) / 2 - 4), rank, fill = 0, font = font)

    # Random rotate transformation
    img = img.rotate(random.uniform(-ROTATE_MAX_ANGLE, ROTATE_MAX_ANGLE), expand = False, fillcolor = '#FFFFFF')
    pixels = list(img.getdata())  # Extract image pixels

    # Introduce random noise
    for (i, _) in enumerate(pixels):
        if random.random() <= noise_level:
            pixels[i] = random.randint(0, 255)  # Replace a chosen pixel with a random intensity

    # Save noisy image
    noisy_img = Image.new('L', img.size)
    noisy_img.putdata(pixels)

    return noisy_img

# print statements to check functionality
if __name__ == "__main__":
    if not os.path.exists(TRAINING_IMAGE_DIR):
        generate_data_set(500, TRAINING_IMAGE_DIR)
        print("Training images generated.")

    if not os.path.exists(TEST_IMAGE_DIR):
        generate_data_set(100, TEST_IMAGE_DIR)
        print("Test images generated.")
