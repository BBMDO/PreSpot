import os
import glob
import argparse
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter
import cv2
from torchvision import transforms
from skimage.segmentation import active_contour
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from skimage import color
from skimage.filters import gaussian, threshold_otsu
from skimage.draw import polygon

# Function to load images from a folder
def load_images_from_folder(folder):
    images = []
    for filename in os.listdir(folder):
        img_path = os.path.join(folder, filename)
        try:
            # Try to open the file as an image
            img = Image.open(img_path)
            img.verify()  # Verify that it is an image
            img = Image.open(img_path)  # Reopen the image after verification
            images.append((filename, img))
        except (IOError, SyntaxError):
            # Skip files that are not images
            print(f"File {filename} is not a valid image.")
    return images

# Function to save an image to a folder
def save_image(image, folder, filename):
    if not os.path.exists(folder):
        os.makedirs(folder)
    image.save(os.path.join(folder, filename))

# Function to apply augmentations to an image
def augment_image(image, augmentations, filename, output_folder, args):
    
    rotation_degree = getattr(args, 'rotation_degree', 15)
    
    if 'r' in augmentations:
        # Rotate image with a specified or default degree
        for angle in range(0, 360, args.rotation_degree):
            rotated_image = image.rotate(angle)
            rotated_filename = f"{filename}_rotated_{angle}.png"
            save_image(rotated_image, output_folder, rotated_filename)

    if 'fh' in augmentations:
        # Apply horizontal flip
        image = image.transpose(Image.FLIP_LEFT_RIGHT)
        filename = f"{filename}_flipped_horizontal.png"
    
    if 'fv' in augmentations:
        # Apply vertical flip
        image = image.transpose(Image.FLIP_TOP_BOTTOM)
        filename = f"{filename}_flipped_vertical.png"
    
    # If multiple augmentations are chosen, combine them
    if 'r' in augmentations and 'fh' in augmentations and 'fv' in augmentations:
        for angle in range(0, 360, args.rotation_degree):
            combined_image = image.rotate(angle).transpose(Image.FLIP_LEFT_RIGHT).transpose(Image.FLIP_TOP_BOTTOM)
            combined_filename = f"{filename}_rotate_{angle}_flip_hv.png"
            save_image(combined_image, output_folder, combined_filename)

    return image  # Return the augmented image

# Function to apply Gabor filter
def apply_gabor_filter(image):
    gabor_kernels = []
    # Create Gabor kernel with different orientations and wavelengths
    for theta in np.arange(0, np.pi, np.pi / 4):
        kernel = cv2.getGaborKernel((21, 21), 8.0, theta, 10.0, 0.5, 0, ktype=cv2.CV_32F)
        gabor_kernels.append(kernel)
    
    # Convert the PIL image to OpenCV format (numpy array)
    image_cv = np.array(image.convert('L'))
    
    # Apply Gabor filters and combine them
    filtered_images = [cv2.filter2D(image_cv, cv2.CV_8UC3, kernel) for kernel in gabor_kernels]
    
    # Combine all filtered images (for simplicity, we average them)
    gabor_image = np.mean(filtered_images, axis=0)
    
    # Convert back to PIL format
    return Image.fromarray(np.uint8(gabor_image))

# Function to apply PCA (Karhunen-Loève Transform)
def apply_pca(image, n_components=2):
    # Convert image to grayscale and then flatten
    gray_image = np.array(image.convert('L'))
    flat_image = gray_image.flatten()

    # Apply PCA
    pca = PCA(n_components=n_components)
    transformed = pca.fit_transform(flat_image.reshape(1, -1))

    # Reshape back into an image
    transformed_image = transformed.reshape(gray_image.shape)
    return Image.fromarray(np.uint8(transformed_image))

# Function to convert image to LAB color space and apply threshold
def segment_using_lab(image):
    # Convert the image to LAB color space
    lab_image = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
    
    # Extract the L channel (lightness)
    l_channel = lab_image[:, :, 0]
    
    # Apply Gaussian smoothing to the L channel
    l_channel_blur = cv2.GaussianBlur(l_channel, (5, 5), 0)
    
    # Apply Otsu's thresholding on the L channel to separate dark and light regions
    _, binary_image = cv2.threshold(l_channel_blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    return binary_image
    
def apply_watershed_simple(image, filename, output_folder):
    # Convert the PIL image to a NumPy array for OpenCV processing
    #image_np = np.array(image)
    en_np_array = np.array(image)
    #Check if the image is already grayscale (1 channel)
    if len(en_np_array.shape) == 2: #The image is already grayscale
        gray = en_np_array #No need to convert
    else:
        #Convert to grayscale if the image is color (3 channels)
        gray = cv2.cvtColor(en_np_array, cv2.COLOR_BGR2GRAY)
    # Enhance contrast
    image_pil = Image.fromarray(en_np_array)
    enhancer = ImageEnhance.Contrast(image_pil)
    en_np = enhancer.enhance(2.0)  # Increase contrast by a factor of 2
    en_np_array = np.array(en_np)

   # Convert to grayscale again after enhancing contrast (if the image is still not grayscale)
    if len(en_np_array.shape) == 3:  # Check if the image is still color
        gray = cv2.cvtColor(en_np_array, cv2.COLOR_BGR2GRAY)

    # Apply a Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (1, 1), 0)

    # Apply Otsu thresholding for binary segmentation
    _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Apply distance transform to highlight the foreground (stain)
    dist_transform = cv2.distanceTransform(binary, cv2.DIST_L2, 5)
    _, sure_fg = cv2.threshold(dist_transform, 0.2 * dist_transform.max(), 255, 0)

    # Dilate the binary image to define the background
    kernel = np.ones((3, 3), np.uint8)
    sure_bg = cv2.dilate(binary, kernel, iterations=1)

    # Convert foreground to uint8
    sure_fg = np.uint8(sure_fg)
    unknown = cv2.subtract(sure_bg, sure_fg)

    # Label connected components for watershed
    _, markers = cv2.connectedComponents(sure_fg)

    # Add 1 to all markers to ensure the background is properly labeled
    markers = markers + 1
    markers[unknown == 255] = 0

    # Convert the image to BGR (CV_8UC3) if it is not already
    if len(en_np_array.shape) == 2: # if the image is grayscale
        en_np_array = cv2.cvtColor(en_np_array, cv2.COLOR_GRAY2BGR)
    
    # Ensure the markers are of ttpe CV_32SC1
    markers = np.int32(markers)

    # Apply the Watershed algorithm
    cv2.watershed(en_np_array, markers)

    # Create a binary image where the interior of the stain is white and the exterior is black
    segmented_image = np.zeros_like(gray)
    segmented_image[markers > 1] = 255  # Interior in white, exterior in black

    # Convert the binary image back to PIL format and save it
    final_image = Image.fromarray(segmented_image)
    save_image(final_image, output_folder, f"{filename}_watershed_simple.png")

# Function to enhance contrast
def enhance_contrast(image, factor=1.5):
	enhancer = ImageEnhance.Contrast(image)
	return enhancer.enhance(2.0)  # Increase contrast by a factor of 2
	    
# Function to sharpen the image
def sharpen_image(image, factor=2.0):
	enhancer = ImageEnhance.Sharpness(image)
	return enhancer.enhance(factor)
	
# Function to apply Gaussian filter
def apply_gaussian_filter(image):
    # Convert the PIL image to OpenCV format (numpy array)
    image_cv = np.array(image)
    
    # Apply Gaussian blur
    blurred_image = cv2.GaussianBlur(image_cv, (5, 5), 0)
    
    # Convert back to PIL format
    return Image.fromarray(blurred_image)
    
# Function to resize an image
def resize_image(image, size):
    return image.resize(size, Image.Resampling.LANCZOS)

# Function to convert an image to grayscale
def convert_to_grayscale(image):
    return image.convert("L")

# Function to add noise to an image
def add_noise(image):
    np_image = np.array(image)
    mean = 0
    std = 0.1
    gauss = np.random.normal(mean, std, np_image.shape)
    noisy_image = np_image + gauss * 255
    noisy_image = np.clip(noisy_image, 0, 255).astype(np.uint8)
    return Image.fromarray(noisy_image)

# Function to normalize an image's pixel values
def normalize_image(image):
    np_image = np.array(image).astype(np.float32) / 255.0
    return Image.fromarray((np_image * 255).astype(np.uint8))

# Function to zoom in or out of an image
def zoom_image(image, zoom_factor, zoom_in=True):
    width, height = image.size
    if zoom_in:
        # Calculate the new dimensions to crop (zoom in)
        crop_width = int(width / zoom_factor)
        crop_height = int(height / zoom_factor)
        left = (width - crop_width) // 2
        top = (height - crop_height) // 2
        right = (width + crop_width) // 2
        bottom = (height + crop_height) // 2
        # Crop the image to zoom in
        image = image.crop((left, top, right, bottom))
    else:
        # Calculate the new dimensions to add borders (zoom out)
        new_width = int(width * zoom_factor)
        new_height = int(height * zoom_factor)
        # Create a new larger image with a black background
        new_image = Image.new("RGB", (new_width, new_height), (0, 0, 0))
        # Calculate the position to paste the original image in the center
        left = (new_width - width) // 2
        top = (new_height - height) // 2
        new_image.paste(image, (left, top))
        image = new_image
    
    return image

# Function to apply shear and translation to an image
def shear_translate_image(image):
    transform = transforms.Compose([
        transforms.RandomAffine(degrees=0, shear=10, translate=(0.1, 0.1)),
        transforms.ToTensor()
    ])
    # Apply the transformations
    transformed_image = transform(image)
    
    # Convert the tensor back to a PIL image
    transformed_image = transforms.ToPILImage()(transformed_image)
    
    return transformed_image

# Function to remove noise from an image using Non-Local Means Denoising
def remove_noise(image):
    np_image = np.array(image)
    blurred = cv2.fastNlMeansDenoisingColored(np_image, None, h=10, templateWindowSize=7, searchWindowSize=21)
    return Image.fromarray(blurred)

# Function to find the high contrast center of an image
def find_high_contrast_center(image):
    #Check if the image is already in grayscale (1 channel)
    if len(np.array(image).shape) == 2: #Grayscale (1 channel)
        gray_image = np.array(image)
    else:
        #Convert to grayscale if the image has 3 channels (RGB)
        gray_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2GRAY)
    
    # Compute the gradients in x and y direction
    grad_x = cv2.Sobel(gray_image, cv2.CV_64F, 1, 0, ksize=5)
    grad_y = cv2.Sobel(gray_image, cv2.CV_64F, 0, 1, ksize=5)
    
    # Compute the gradient magnitude
    grad_mag = cv2.magnitude(grad_x, grad_y)
    
    # Find the location of the maximum gradient magnitude
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(grad_mag)
    
    # Return the coordinates of the highest contrast point
    return max_loc

# Function to remove hair from an image
def remove_hair(image, 
                median_filter_size=3, 
                scaling_factor=1.64, 
                hair_threshold=20, 
                kernel_size_small=(3, 3), 
                kernel_size_large=(7, 7), 
                kernel_dilate_size=(5, 5), 
                inpainting_radius=25, 
                canny_threshold1=30, 
                canny_threshold2=100, 
                ruler_kernel_size=(2, 50)):
    
    # Convert the PIL image to a NumPy array
    np_image = np.array(image)

    # Convert the image to the red grayscale channel (Ired)
    red_channel = np_image[:, :, 2]  # Extract the red channel

    # Apply a median filter with the specified kernel size to smooth the image and reduce noise
    median_filtered = cv2.medianBlur(red_channel, median_filter_size)

    # Linearly scale the resultant image by the specified scaling factor
    scaled_red_channel = cv2.convertScaleAbs(median_filtered, alpha=scaling_factor)

    # Initial hair detection using ROI-based horizontal and vertical filters
    def detect_hair_roi(img, horizontal=True):
        hair_mask = np.zeros_like(img)
        if horizontal:
            for y in range(img.shape[0]):
                for x in range(img.shape[1] - 6):  # Horizontal ROI of 1x7 pixels
                    roi = img[y, x:x+7]
                    if np.ptp(roi) > 10:  # Check intensity range
                        hair_pixel = np.argmin(roi) + x
                        hair_mask[y, hair_pixel] = 255
        else:
            for x in range(img.shape[1]):
                for y in range(img.shape[0] - 6):  # Vertical ROI of 1x7 pixels
                    roi = img[y:y+7, x]
                    if np.ptp(roi) > 10:  # Check intensity range
                        hair_pixel = np.argmin(roi) + y
                        hair_mask[hair_pixel, x] = 255
        return hair_mask

    # Horizontal and vertical hair detection using ROI masks
    hair_mask_horizontal = detect_hair_roi(scaled_red_channel, horizontal=True)
    hair_mask_vertical = detect_hair_roi(scaled_red_channel, horizontal=False)

    # Combine the hair masks to enhance the hair detection
    combined_hair_mask = np.maximum(hair_mask_horizontal, hair_mask_vertical)

    # Ruler Mark Detection
    def detect_ruler_marks(img):
        red_channel = img[:, :, 2]
        kernel_rect = cv2.getStructuringElement(cv2.MORPH_RECT, ruler_kernel_size)
        red_closed = cv2.morphologyEx(red_channel, cv2.MORPH_CLOSE, kernel_rect)
        ruler_enhanced = cv2.subtract(red_closed, red_channel)
        tr = 2 / 5 * (np.max(ruler_enhanced) - np.min(ruler_enhanced)) + np.min(ruler_enhanced)
        _, ruler_mask_binary = cv2.threshold(ruler_enhanced, tr, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(ruler_mask_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            if cv2.contourArea(contour) < 25 or cv2.isContourConvex(contour) or cv2.boundingRect(contour)[3] < 50:
                cv2.drawContours(ruler_mask_binary, [contour], -1, 0, -1)
        return ruler_mask_binary

    ruler_mask = detect_ruler_marks(np_image)
    combined_hair_mask = cv2.bitwise_or(combined_hair_mask, ruler_mask)

    # White Hair Detection
    def detect_white_hair(img):
        blue_channel = img[:, :, 0]
        median_filtered_blue = cv2.medianBlur(blue_channel, 3)
        white_hair_enhanced = cv2.subtract(blue_channel, median_filtered_blue)
        _, white_hair_mask = cv2.threshold(white_hair_enhanced, 1, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(white_hair_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            if cv2.contourArea(contour) < 20 or cv2.boundingRect(contour)[2] < 20:
                cv2.drawContours(white_hair_mask, [contour], -1, 0, -1)
        return white_hair_mask

    white_hair_mask = detect_white_hair(np_image)
    combined_hair_mask = cv2.bitwise_or(combined_hair_mask, white_hair_mask)

    # Hair Enhancement using background subtraction
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    background = cv2.morphologyEx(combined_hair_mask, cv2.MORPH_CLOSE, kernel)
    enhanced_hair = cv2.subtract(background, scaled_red_channel)

    # Hair Segmentation using gradient-based edge detection in three orientations
    sobelx = cv2.Sobel(enhanced_hair, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(enhanced_hair, cv2.CV_64F, 0, 1, ksize=3)
    sobel_combined = cv2.sqrt(sobelx**2 + sobely**2)
    sobel_combined = cv2.convertScaleAbs(sobel_combined)

    # Apply adaptive thresholding to create a binary mask of hair regions
    _, hair_mask_binary = cv2.threshold(sobel_combined, hair_threshold, 255, cv2.THRESH_BINARY)

    # Use edge detection to capture finer hair strands
    edges_canny = cv2.Canny(sobel_combined, canny_threshold1, canny_threshold2)
    hair_mask_binary = cv2.bitwise_or(hair_mask_binary, edges_canny)

    # Refine the mask using morphological operations
    kernel_refine = np.ones(kernel_size_large, np.uint8)
    hair_mask_binary = cv2.morphologyEx(hair_mask_binary, cv2.MORPH_CLOSE, kernel_refine)
    hair_mask_binary = cv2.morphologyEx(hair_mask_binary, cv2.MORPH_OPEN, kernel_refine)

    # Apply GaussianBlur to smooth the mask for better inpainting
    hair_mask_binary = cv2.GaussianBlur(hair_mask_binary, (9, 9), 0)
    
    # Dilate the hair mask to ensure coverage of all hair regions before inpainting
    kernel_dilate = np.ones(kernel_dilate_size, np.uint8)
    hair_mask_binary = cv2.dilate(hair_mask_binary, kernel_dilate, iterations=1)

    # Use advanced inpainting techniques with the specified radius
    inpainted = cv2.inpaint(np_image, hair_mask_binary, inpainting_radius, cv2.INPAINT_TELEA)

    # Convert the resulting array back to a PIL image
    result_image = Image.fromarray(inpainted)

    return result_image
    
#Process_images function
def process_images(input_folder, output_folder, args):
    images = load_images_from_folder(input_folder)
    for filename, image in images:
        original_filename = filename
        filename = f"processed_{filename}"
        
        # Find the high contrast center
        high_contrast_center = find_high_contrast_center(image)
        print(f"High contrast center for {original_filename}: {high_contrast_center}")

 		# Apply contrast improvement
        if args.contrast:
       		image = enhance_contrast(image, factor=1.8)  # Adjust the factor as needed

        # Apply sharpness improvement
        if args.sharpness:
        	image = sharpen_image(image, factor=2.5)  # Adjust the factor as needed
        
        # Apply resizing, color conversion, and other transformations as needed
        if args.resize:
            width, height = map(int, args.resize.split('x'))
            image = resize_image(image, (width, height))
        
        # Apply Gabor filter if specified
        if args.gabor:
            image = apply_gabor_filter(image)
             
        # Apply Watershed Technique if specified
        if args.border:
            apply_watershed_simple(image, filename, output_folder)
            jpg_files = glob.glob(os.path.join(output_folder, f"{filename.split('_')[0]}*.jpg"))
            for jpg_file in jpg_files:
                os.remove(jpg_file)
        	
        # Apply Gaussian filter if specified
        if args.gaussian:
            image = apply_gaussian_filter(image)
            
        # Apply hair removal if specified
        if args.remove_hair and args.color == 'gray':
            image = remove_hair(image)
        
        # Apply color conversion
        if args.color:
            if args.color == 'gray':
                image = convert_to_grayscale(image)
            elif args.color == 'rgb':
                image = image.convert("RGB")
                        
        if args.augment:
            image = augment_image(image, args.augment, original_filename, output_folder, args)
        
        if args.zoom_in:
            image = zoom_image(image, args.zoom_in, zoom_in=True)
        
#        if args.zoom_out:
#            image = zoom_image(image, args.zoom_out, zoom_in=False)
        
        if args.normalize:
            image = normalize_image(image)
        
        if args.shear_translate:
            image = shear_translate_image(image)
        
        if args.remove_noise:
            image = remove_noise(image)
        
        # Save the processed image
        save_image(image, output_folder, filename)

# Modifications in the main function to include the new hair removal option
def main():
    parser = argparse.ArgumentParser(description='Process images for deep learning.')
    parser.add_argument('-f', '--folder', required=True, help='Input folder containing images.')
    parser.add_argument('-o', '--output', required=True, help='Output folder for processed images.')
    parser.add_argument('-r', '--resize', help='Resize images to WIDTHxHEIGHT format.')
    parser.add_argument('-c', '--color', choices=['gray', 'rgb'], help='Convert images to grayscale or RGB.')
    parser.add_argument('-ag', '--augment', type=str, help='Augmentation options: r (rotate), fh (flip horizontal), fv (flip vertical).')
    parser.add_argument('-rg', '--rotation_degree', type=int, default=15, help='Degree for rotation, default is 15')
    parser.add_argument('-zi', '--zoom_in', type=float, help='Zoom in factor.')
#   parser.add_argument('-zo', '--zoom_out', type=float, help='Zoom out factor.')
    parser.add_argument('-n', '--normalize', action='store_true', help='Normalize pixel values.')
    parser.add_argument('-st', '--shear_translate', action='store_true', help='Apply shear and translation.')
    parser.add_argument('--remove_noise', action='store_true', help='Remove noise from images.')
    parser.add_argument('--remove_hair', action='store_true', help='Remove hair from images.')  # New option to remove hair
    parser.add_argument('-g', '--gabor', action='store_true', help='Apply Gabor filter to the images.')
    parser.add_argument('-b', '--border', action='store_true', help='Apply Watershed Technique.')
    parser.add_argument('-ga', '--gaussian', action='store_true', help='Apply Gaussian filter.')
    parser.add_argument('-co', '--contrast', action='store_true', help='Apply Constrast filter.')
    parser.add_argument('-sh', '--sharpness', action='store_true', help='Apply Sharpness filter.')
   
    args = parser.parse_args()
    
    process_images(args.folder, args.output, args)

if __name__ == "__main__":
    main()