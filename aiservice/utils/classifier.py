import numpy as np
from PIL import Image
import io

def validate_image(image_bytes):
    """
    Simple image validation without using incompatible models.
    Checks if the image can be opened and has reasonable dimensions.
    """
    try:
        # Try to open the image
        image = Image.open(image_bytes)
        
        # Check image size
        width, height = image.size
        
        # Image must be at least 50x50 pixels
        if width < 50 or height < 50:
            print(f"Image too small: {width}x{height}")
            return False
        
        # Image should not be extremely large (e.g., > 4000x4000)
        if width > 4000 or height > 4000:
            print(f"Image too large: {width}x{height}")
            return False
        
        # Check if image has valid color mode
        if image.mode not in ['RGB', 'RGBA', 'L', 'P']:
            print(f"Invalid color mode: {image.mode}")
            return False
        
        print(f"Image validation successful: {width}x{height}, mode: {image.mode}")
        return True
        
    except Exception as e:
        print(f"Image validation error: {str(e)}")
        return False
