import numpy as np
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt


def load_and_wrap():
    """
    Loads and formats MNIST data using Keras

    Returns:
    A tuple containing (training_data, validation_data, test_data).
        - training_data: A list of 50,000 tuples (x, y).
          'x' is a (784, 1) numpy.ndarray with the image data.
          'y' is a (10, 1) numpy.ndarray (one-hot encoded).

        - validation_data: A list of 10,000 tuples (x, y).
          'x' is a (784, 1) numpy.ndarray with the image data.
          'y' is the integer label (0-9).

        - test_data: A list of 10,000 tuples (x, y).
          'x' is a (784, 1) numpy.ndarray with the image data.
          'y' is the integer label (0-9).
    """

    # Keras loads the data as:
    # (training_images, training_labels), (test_images, test_labels)
    # It automatically handles downloading if the data isn't found.
    (tr_images, tr_labels), (te_images, te_labels) = mnist.load_data()

    # --- Create Training Data ---
    # Normalize and reshape the training images
    tr_images = tr_images / 255.0  # Normalize pixel values
    # poo=1421
    # plt.imshow(tr_images[poo], cmap='gray')
    # plt.show()
    # print(tr_labels[poo])
    training_inputs = [np.reshape(x, (784, 1)) for x in tr_images]

    # One-hot encode the training labels
    training_results = [np.reshape(to_categorical(y, num_classes=10), (10, 1))
                        for y in tr_labels]

    # We need to split the Keras training data to create a validation set.
    # The original script had 50,000 training and 10,000 validation images.
    # Keras provides 60,000 training images, so we'll split them.
    final_training_inputs = training_inputs[:50000]
    final_training_results = training_results[:50000]
    training_data = list(zip(final_training_inputs, final_training_results))

    val_inputs = training_inputs[50000:]
    val_labels = tr_labels[50000:]  # Use original int labels for validation
    validation_data = list(zip(val_inputs, val_labels))

    # --- Create Test Data ---
    # Normalize and reshape the test images
    te_images = te_images / 255.0  # Normalize pixel values
    test_inputs = [np.reshape(x, (784, 1)) for x in te_images]
    test_data = list(zip(test_inputs, te_labels))  # Use original int labels

    return (training_data, validation_data, test_data)
