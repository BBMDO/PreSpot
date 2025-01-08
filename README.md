# PreSpot
Image Processing System for Melanoma Diagnosis

## Introduction
This Python program was designed to perform transformations and improvements on a batch of images, initially for melanoma, but later other applications were explored, such as retina, X-rays, MRIs, and ultrasounds. It is ideal for melanoma and may require fine adjustments for other examples depending on the specific image customization needs. It can perform resizing, color conversion, augmentation (such as rotation and flipping), noise reduction, filtering, and more advanced image processing techniques like applying Gabor and Gaussian filters. This program is ideal for preparing images for machine learning applications or for performing general image preprocessing tasks.

## Installation and Dependencies
This script requires Python 3.x and several Python libraries for image processing. You can install these dependencies with the following command:
```bash
pip install numpy pillow opencv-python
```

## Usage and Arguments
The program accepts several arguments, making it highly customizable. Below is a description of each available argument:

1. -f, --folder: Path to the input folder containing images.
   - Required: Yes
   - Example: `-f ./input_images`

2. -o, --output: Path to the output folder where processed images will be saved.
   - Required: Yes
   - Example: `-o ./output_images`

3. -r, --resize: Resize images to a specified format (WIDTHxHEIGHT).
   - Example: `-r 224x224`

4. -c, --color: Convert images to grayscale or RGB.
   - Choices: `gray`, `rgb`
   - Example: `-c gray`

5. -ag, --augment: Perform image augmentation. Options include:
   - `r` for rotation
   - `fh` for horizontal flip
   - `fv` for vertical flip
   - Example: `-ag r,fv`

6. -rg, --rotation_degree: Degree of rotation for augmentation, with a default of 15.
   - Example: `-rg 30`

7. -zi, --zoom_in: Apply zoom-in transformation by a specified factor.
   - Example: `-zi 1.5`

8. -zo, --zoom_out: Apply zoom-out transformation by a specified factor.
   - Example: `-zo 0.8`

9. -n, --normalize: Normalize pixel values across all images.
   - Action: `store_true`
   - Example: `-n`

10. -st, --shear_translate: Apply shear and translation transformation.
    - Action: `store_true`
    - Example: `-st`

11. --remove_noise: Reduce noise in the images.
    - Action: `store_true`
    - Example: `--remove_noise`

12. --remove_hair: Remove hair-like artifacts from images (useful for skin images).
    - Action: `store_true`
    - Example: `--remove_hair`

13. -g, --gabor: Apply the Gabor filter to enhance textures.
    - Action: `store_true`
    - Example: `-g`

14. -b, --border: Apply the Watershed Technique for border detection.
    - Action: `store_true`
    - Example: `-b`

15. -ga, --gaussian: Apply a Gaussian filter for blurring or noise reduction.
    - Action: `store_true`
    - Example: `-ga`

16. -co, --contrast: Adjust the contrast of the images.
    - Action: `store_true`
    - Example: `-co`

17. -sh, --sharpness: Apply a sharpness filter to enhance edges.
    - Action: `store_true`
    - Example: `-sh`

## Running the Program
To run the program, use the following command format:
```bash
python image_resolution_v8_modularized.py -f PATH_TO_INPUT_FOLDER -o PATH_TO_OUTPUT_FOLDER [OPTIONS]
```
Replace `PATH_TO_INPUT_FOLDER` and `PATH_TO_OUTPUT_FOLDER` with the desired input and output folder paths, and add any optional arguments as needed.

### Example Usage
Below are a few examples to illustrate common use cases:

1. Resize images and convert to grayscale:
   ```bash
   python image_resolution_v8_modularized.py -f ./input_images -o ./output_images -r 224x224 -c gray
   ```

2. Apply augmentation with rotation and flipping, normalize pixel values:
   ```bash
   python image_resolution_v8_modularized.py -f ./input_images -o ./output_images -ag r,fh -n
   ```

3. Apply Gabor filter, contrast adjustment, and sharpen images:
   ```bash
   python image_resolution_v8_modularized.py -f ./input_images -o ./output_images -g -co -sh
   ```

## Notes
- Order of Operations: The transformations are applied sequentially based on the order in which the arguments are given.
- File Saving: Processed images are saved in the specified output folder, with filenames indicating the transformations applied.

## Troubleshooting
If you encounter issues, verify that all dependencies are correctly installed and compatible with your Python version.


