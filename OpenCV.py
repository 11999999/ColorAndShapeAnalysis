import cv2
import numpy as np
from scipy.stats import skew
from skimage.measure import label, regionprops
from skimage import measure
from sklearn.cluster import KMeans
from tabulate import tabulate

def read_image(image_path):
    """Reads an image from a given path and returns it.
    
    Args:
        image_path (str): Path to the image file.

    Returns:
        numpy.ndarray: The image read from the given path.

    Raises:
        ValueError: If the image cannot be found or read.
    """
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError("Image not found or invalid image path.")
    return image

def detect_primary_colors(image, colors):
    """Detects and prints primary colors present in the image.
    
    Args:
        image (numpy.ndarray): The image in which to detect colors.
        colors (dict): A dictionary of color names and their BGR ranges.
    """
    present_colors = []
    for color, (lower, upper) in colors.items():
        lower_bound = np.array(lower, dtype="uint8")
        upper_bound = np.array(upper, dtype="uint8")
        mask = cv2.inRange(image, lower_bound, upper_bound)
        if np.any(mask):
            present_colors.append(color)
    print("Detected colors:", ", ".join(present_colors))

def segment_image_kmeans(image, num_colors=3):
    """Segments the image into different objects according to their colors using K-means.
    
    Args:
        image (numpy.ndarray): The input color image.
        num_colors (int): The number of color clusters to segment the image into.

    Returns:
        numpy.ndarray: The image segmented into `num_colors` colors.
    """
    reshaped_image = image.reshape((-1, 3)).astype(np.float32)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
    _, labels, centers = cv2.kmeans(reshaped_image, num_colors, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    segmented_image = centers[labels.flatten()].reshape(image.shape).astype(np.uint8)
    return segmented_image

def color_slicing(image, target_color, width=20):
    """Isolates a specific color within a width around the target color.
    
    Args:
        image (numpy.ndarray): The input color image.
        target_color (tuple): The BGR color to isolate.
        width (int): The width around the target color to include.

    Returns:
        numpy.ndarray: A mask of the isolated color area.
    """
    lower_bound = np.array([max(0, tc - width // 2) for tc in target_color], dtype="uint8")
    upper_bound = np.array([min(255, tc + width // 2) for tc in target_color], dtype="uint8")
    mask = cv2.inRange(image, lower_bound, upper_bound)
    return mask

def convert_to_grayscale(image):
    """Converts a color image to grayscale.
    
    Args:
        image (numpy.ndarray): The input color image.

    Returns:
        numpy.ndarray: The grayscale image.
    """
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

def calc_histogram_features(image):
    """Calculates histogram features of an image.
    
    Args:
        image (numpy.ndarray): The input image.

    Returns:
        dict: A dictionary of histogram features.
    """
    histogram = cv2.calcHist([image], [0], None, [256], [0, 256])
    features = {
        'mean': np.mean(histogram),
        'std_dev': np.std(histogram),
        'skewness': skew(histogram)[0],
        'energy': np.sum(histogram**2),
        'entropy': measure.shannon_entropy(image)
    }
    return features

def print_histogram_features(features):
    """Prints histogram features in a formatted table.
    
    Args:
        features (dict): The histogram features to print.
    """
    print("\nGrayscale Image Histogram Features:")
    print(tabulate([features], headers="keys", tablefmt="pretty"))

def process_image(image_path):
    """Processes an image for color detection, segmentation, and feature calculation.
    
    Args:
        image_path (str): Path to the image file.
    """
    color_image = read_image(image_path)
    primary_colors = {
        'Red': ((0, 0, 128), (50, 50, 255)),
        # Define other colors as needed
    }
    detect_primary_colors(color_image, primary_colors)
    segmented_image = segment_image_kmeans(color_image, num_colors=5)
    sliced_mask = color_slicing(color_image, (128, 128, 255), width=20)
    gray_image = convert_to_grayscale(color_image)
    hist_features = calc_histogram_features(gray_image)
    print_histogram_features(hist_features)

if __name__ == "__main__":
    image_path = 'image.jpg' # Update this path to your image file
    process_image(image_path)
