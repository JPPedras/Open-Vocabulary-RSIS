import numpy as np
from scipy.ndimage import label as lb
import math




def find_clusters(mask):

    labels, num_clusters = lb(mask)
    cluster_labels = np.zeros_like(labels)
    for i in range(1, num_clusters+1):
        cluster_labels[labels == i] = i

    return cluster_labels


def find_big_cluster(clusters):
    unique_labels = np.unique(clusters)
    sizes = []
    for label in unique_labels:
        if label == 0:
            continue  # skip background label
        size = np.sum(clusters == label)
        sizes.append(size)
    sorted_indices = np.argsort(sizes)[::-1]  # get indices in descending order
    sorted_sizes = np.array(sizes)[sorted_indices]
    if len(sorted_sizes) >= 2 and sorted_sizes[0] >= sorted_sizes[1] * 2:
        return unique_labels[sorted_indices[0]]+1
    else:
        return None



def generate_crops(image, annotation):

    # Get the width and height of the input image
    width, height = image.size

    crop_size=672
    num_crops = 3
    max_angle = 90

    # Generate the crops
    image_crops = []
    annotation_crops = []

    if width == 1024 and height == 1024:
        for i in range(num_crops):
            # Choose a random size between min_size and max_size
            size = crop_size
            # Choose a random rotation angle in degrees
            angle = np.random.randint(-max_angle, max_angle + 1)
            # Calculate the maximum allowable distance between the center of the crop and the edge of the image
            angle_rad = np.deg2rad(angle)
            #max_allowable_distance = int(max(size / 2 * abs(np.cos(angle_rad)), size / 2 * abs(np.sin(angle_rad))))
            max_allowable_distance = int(size/math.sqrt(2))
            # Choose a random location for the center of the crop
            x_center = np.random.randint(max_allowable_distance, width - max_allowable_distance)
            y_center = np.random.randint(max_allowable_distance, height - max_allowable_distance)
            # Define the coordinates of the corners of the rotated crop
            x1, y1 = int(x_center - (size // 2)), int(y_center - (size // 2))
            x2, y2 = int(x_center + (size // 2)), int(y_center + (size // 2))
            # Rotate the image and annotation
            crop1 = image.rotate(angle).crop((x1, y1, x2, y2))
            crop2 = annotation.rotate(angle).crop((x1, y1, x2, y2))
            # Add the crop to the list of crops
            image_crops.append(crop1)
            annotation_crops.append(crop2)

    return image_crops, annotation_crops


def get_side_masks(mask):

    half = mask.shape[0] // 2

    left_half = mask[:, :half]
    right_half = mask[:, half:]
    left_labels = list(np.unique(left_half[left_half > 0]))
    right_labels = list(np.unique(right_half[right_half > 0]))

    left_mask = np.where(np.isin(mask, right_labels+[0]), 0, mask)
    right_mask = np.where(np.isin(mask, left_labels+[0]), 0, mask)

    return left_mask, right_mask

