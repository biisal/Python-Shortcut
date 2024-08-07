# pip install opencv-python-headless pillow numpy

import cv2
import numpy as np
from PIL import Image, ImageEnhance

def enhance_image(input_image_path, output_image_path):
    # Load the image
    image = cv2.imread(input_image_path)
    if image is None:
        print(f"Error loading image at {input_image_path}")
        return
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(image_rgb)
    enhancer = ImageEnhance.Sharpness(pil_image)
    sharp_image = enhancer.enhance(2.0)  # Increase sharpness by a factor of 2
    enhancer = ImageEnhance.Contrast(sharp_image)
    contrast_image = enhancer.enhance(1.5)  # Increase contrast by a factor of 1.5
    enhancer = ImageEnhance.Brightness(contrast_image)
    bright_image = enhancer.enhance(1.2)  # Increase brightness by a factor of 1.2
    enhanced_image = np.array(bright_image)
    enhanced_image = cv2.cvtColor(enhanced_image, cv2.COLOR_RGB2BGR)
    denoised_image = cv2.fastNlMeansDenoisingColored(enhanced_image, None, 10, 10, 7, 21)
    cv2.imwrite(output_image_path, denoised_image)
    print(f"Enhanced image saved at {output_image_path}")


input_image_path = 'input.jpg'
output_image_path = 'enhanced_output.jpg'
enhance_image(input_image_path, output_image_path)
