#%%
import sys
print(sys.executable)
#%%
import cv2
import numpy as np
import os
from skimage.filters import threshold_otsu
from skimage.filters import gabor
import matplotlib.pyplot as plt
from skimage.filters import threshold_otsu

#%%
os.chdir(r'H:\Thesis')

#%%

def count_total_bounding_boxes(labels_folder):
    total_bounding_boxes = 0
    label_files = [f for f in os.listdir(labels_folder) if f.endswith('.txt')]

    for label_file in label_files:
        label_file_path = os.path.join(labels_folder, label_file)
        with open(label_file_path, 'r') as file:
            lines = file.readlines()
            total_bounding_boxes += len(lines)  # Assuming each line in a label file represents a bounding box

    return total_bounding_boxes  # Corrected variable name here


# Paths to your labels folders
labels_folders = [
    r'./roboflowV6_1169\train\labels',
    r'./roboflowV6_1169\valid\labels',
    r'./roboflowV6_1169\test\labels'
]

total_bboxes = 0
for folder in labels_folders:
    bboxes = count_total_bounding_boxes(folder)
    total_bboxes += bboxes
    print(f"Total bounding boxes in {folder}: {bboxes}")

print(f"Grand total of bounding boxes across all folders: {total_bboxes}")


#%%

# Function to read bounding box coordinates from label file
def read_label_file(label_file_path):
    with open(label_file_path, 'r') as file:
        lines = file.readlines()
        # Extract bounding box coordinates
        bbox_coords = []
        for line in lines:
            values = line.strip().split()
            # Extract the last four values (x, y, width, height)
            bbox_coords.append(list(map(float, values[1:5])))
    return bbox_coords

# Function to convert normalized bounding box coordinates to pixel values


def convert_yolov9_to_pixel(bbox_yolov9, image_width, image_height):
    centerX, centerY, width, height = bbox_yolov9
    
    # Calculate bounding box coordinates in pixel values
    x_center_pixel = int(centerX * image_width)
    y_center_pixel = int(centerY * image_height)
    width_pixel = int(width * image_width)
    height_pixel = int(height * image_height)
    
    # Calculate x_min, y_min, x_max, y_max
    x_min = x_center_pixel - (width_pixel // 2)
    y_min = y_center_pixel - (height_pixel // 2)
    x_max = x_min + width_pixel
    y_max = y_min + height_pixel
    
    # Return bounding box coordinates as tuple
    bbox_pixel = (x_min, y_min, x_max, y_max)
    
    return bbox_pixel
# Function to crop bounding boxes from images
def crop_bounding_boxes(images_folder, labels_folder, output_folder):
    # Create output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # List all image files
    image_files = os.listdir(images_folder)

    for image_file in image_files:
        if image_file.endswith('.jpg'):  # Assuming images are JPEG format
            # Get image path
            image_path = os.path.join(images_folder, image_file)
            # Read image
            image = cv2.imread(image_path)

            # Get corresponding label file path
            label_file_name = image_file.replace('.jpg', '.txt')
            label_file_path = os.path.join(labels_folder, label_file_name)

            # Read bounding box coordinates from label file
            bbox_coords = read_label_file(label_file_path)

            # Crop bounding boxes from image and save
            for i, bbox_normalized in enumerate(bbox_coords):
                # Convert normalized coordinates to pixel values
                bbox_pixel = convert_yolov9_to_pixel(bbox_normalized, image.shape[1], image.shape[0])
                x_min, y_min, x_max, y_max = bbox_pixel

                # Crop ROI from image
                cropped_image = image[y_min:y_max, x_min:x_max]

                # Save cropped image with unique name for each bounding box
                output_file_name = f'{os.path.splitext(image_file)[0]}_bbox_{i}.jpg'
                output_file_path = os.path.join(output_folder, output_file_name)
                cv2.imwrite(output_file_path, cropped_image)

                print(f'Cropped bounding box {i} from {image_file} saved as {output_file_name}')

# Define the paths for each dataset
datasets = {
    "train": {
        "images_folder": r'./roboflowV6_1169\train\images',
        "labels_folder": r'./roboflowV6_1169\train\labels',
        "cropped_folder": './train_cropped_mandarins'
    },
    "valid": {
        "images_folder": r'./roboflowV6_1169\valid\images',
        "labels_folder": r'./roboflowV6_1169\valid\labels',
        "cropped_folder": './valid_cropped_mandarins'
    },
    "test": {
        "images_folder": r'./roboflowV6_1169\test\images',
        "labels_folder": r'./roboflowV6_1169\test\labels',
        "cropped_folder": './test_cropped_mandarins'
    }
}

# Process each dataset
for key, paths in datasets.items():
    crop_bounding_boxes(paths["images_folder"], paths["labels_folder"], paths["cropped_folder"])

#%%
#chosen segmentation method
#pre processing before segmentation
def preprocessing_before_threshold_segmentation(image):
     # Convert from RGB to HSL
     hsl_image = cv2.cvtColor(image, cv2.COLOR_BGR2HLS)

     # Split into separate channels
     h, l, s = cv2.split(hsl_image)

     # Apply CLAHE to lightness channel
     clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
     l_clahe = clahe.apply(l)

     # Merge the HSL channels back together
     hsl_clahe = cv2.merge((h, l_clahe, s))

     # Convert back to RGB
     rgb_clahe = cv2.cvtColor(hsl_clahe, cv2.COLOR_HLS2BGR)

     # Apply Gaussian blur to reduce noise
     blurred_image = cv2.GaussianBlur(rgb_clahe, (5, 5), 0)
     
     # Convert the image to grayscale
     gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
     # Apply Canny edge detection
     edges = cv2.Canny(gray_image, 100, 200)

     frequency = 0.3
     theta = 0.6
     sigma = 2.3
     # Apply Gabor filter
     gabor_response, _ = gabor(gray_image, frequency, theta=theta, sigma_x=sigma, sigma_y=sigma)
     
     # Normalize the Gabor response
     gabor_response_normalized = cv2.normalize(gabor_response, None, 0, 255, cv2.NORM_MINMAX)
     # Combine edges with Gabor response
     combined_image = cv2.bitwise_or(edges, gabor_response_normalized)
     return combined_image
 

#segmentataion with gabor and canny
def itiration_images_gabor(images_folder, output_directory):
    # Create output folder if it doesn't exist
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)
    # List all image files
    image_files = os.listdir(images_folder)
    for image_file in image_files:
        if image_file.endswith('.jpg'):  # Assuming images are JPEG format
            # Get image path
            image_path = os.path.join(images_folder, image_file)
            # Preprocess the image
            image = cv2.imread(image_path)
            segmented_image= preprocessing_before_threshold_segmentation(image)
            # Perform morphological operations to refine the segmented shape
            kernel = np.ones((9, 9), np.uint8)
            segmented_image = cv2.morphologyEx(segmented_image, cv2.MORPH_CLOSE, kernel)
            output_path = os.path.join(output_directory, image_file)
            cv2.imwrite(output_path, cv2.cvtColor(segmented_image, cv2.COLOR_RGB2BGR))                
            print('Segmented gabor saved')
    return


# Define input and output folders
output_folder = './segmented_with_gabor_mandarins'

# Perform segmentation with Gabor filter
itiration_images_gabor(cropped_folder, output_folder)


#%%
#kmeans segmentation- we ended up not choosing this method.

def apply_kmeans_color_clustering(image, clusters_count):
    # Convert the image to 'number of pixels' x BGR
    image_BGR = np.float32(image.reshape((-1, 3)))
    # Define criteria, number of clusters (K), and apply kmeans
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    ret, labels, centers = cv2.kmeans(image_BGR, clusters_count, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    # Convert back into uint8
    centers = np.uint8(centers)
    # Make the original image with center colors
    clustered_image = centers[labels.flatten()]
    # Reshape it back to the shape of the original image
    clustered_image = clustered_image.reshape(image.shape)
    return clustered_image


from skimage.filters import threshold_otsu
def itiration_images_kmeans(images_folder, output_directory):
    # Create output folder if it doesn't exist
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)
    # List all image files
    image_files = os.listdir(images_folder)
    for image_file in image_files:
        if image_file.endswith('.jpg'):  # Assuming images are JPEG format
            # Get image path
            image_path = os.path.join(images_folder, image_file)
            # Preprocess the image
            image = cv2.imread(image_path)
            #Apply Otsu Thresholding on Image
            my_image_color_RGB = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            # Apply KMeans color clustering
            my_image_clusters_color_2 = apply_kmeans_color_clustering(my_image_color_RGB, 2)
            
            output_path = os.path.join(output_directory, image_file)
            cv2.imwrite(output_path, cv2.cvtColor(my_image_clusters_color_2, cv2.COLOR_RGB2BGR))                
            print('Segmented Otsu saved')
    return

output_folder = './segmented_Kmeans_mandarins'
itiration_images_kmeans(cropped_folder, output_folder)


#%%Otsu segmentation- e ended up not choosing this method.

#segmentation
def filter_image(image, mask):

    r = image[:,:,0] * mask
    g = image[:,:,1] * mask
    b = image[:,:,2] * mask

    return np.dstack([r,g,b])


from skimage.filters import threshold_otsu
def itiration_images_otsu(images_folder, output_directory):
    # Create output folder if it doesn't exist
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)
    # List all image files
    image_files = os.listdir(images_folder)
    for image_file in image_files:
        if image_file.endswith('.jpg'):  # Assuming images are JPEG format
            # Get image path
            image_path = os.path.join(images_folder, image_file)
            # Preprocess the image
            image = cv2.imread(image_path)
            #Apply Otsu Thresholding on Image
            img_gray=cv2.cvtColor(image,cv2.COLOR_RGB2GRAY)
            thresh = threshold_otsu(img_gray)
            img_otsu  = img_gray < thresh
            filtered = filter_image(image, img_otsu)
            output_path = os.path.join(output_directory, image_file)
            cv2.imwrite(output_path, cv2.cvtColor(filtered, cv2.COLOR_RGB2BGR))                
            print('Segmented Otsu saved')
    return

output_folder = './segmented_Otsu_mandarins'
itiration_images_otsu(cropped_folder, output_folder)

#%%watershed segmentation- we ended up not choosing this method.
def apply_watershed_with_threshold(img, threshold_value=None):
    # Read the image
    my_image_color = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Use Otsu's method if threshold_value is not specified
    if threshold_value is None:
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    else:
        _, thresh = cv2.threshold(gray, threshold_value, 255, cv2.THRESH_BINARY)

    # Noise removal
    kernel = np.ones((3, 3), np.uint8)
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)

    # Sure background area
    sure_bg = cv2.dilate(opening, kernel, iterations=3)

    # Finding sure foreground area
    dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
    _, sure_fg = cv2.threshold(dist_transform, 0.7 * dist_transform.max(), 255, 0)

    # Finding unknown region
    sure_fg = np.uint8(sure_fg)
    unknown = cv2.subtract(sure_bg, sure_fg)

    # Marker labeling
    _, markers = cv2.connectedComponents(sure_fg)

    # Add one to all labels so that sure background is not 0, but 1
    markers = markers + 1

    # Now, mark the region of unknown with zero
    markers[unknown == 255] = 0

    # Apply Watershed
    markers = cv2.watershed(img, markers)
    img[markers == -1] = [255, 0, 0]
    return markers

def itiration_images_watershed(images_folder, output_directory):
    # Create output folder if it doesn't exist
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)
    # List all image files
    image_files = os.listdir(images_folder)
    for image_file in image_files:
        if image_file.endswith('.jpg'):  # Assuming images are JPEG format
            # Get image path
            image_path = os.path.join(images_folder, image_file)
            # Preprocess the image
            image = cv2.imread(image_path)
            # Preprocess the image
            watershed = apply_watershed_with_threshold(image, threshold_value=None)
            # Save the segmented image
            output_path = os.path.join(output_directory, image_file)
            cv2.imwrite(output_path, watershed)
            print('Segmented watershed saved:', output_path)
    return

output_folder = './segmented_watershed_mandarins'
itiration_images_watershed(cropped_folder, output_folder)

#the watershed results are not very helpful