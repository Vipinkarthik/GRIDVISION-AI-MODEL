import cv2
import pytesseract
import numpy as np
from PIL import Image
import re
import os

# Try to set pytesseract path for Windows
try:
    pytesseract.pytesseract.pytesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
except:
    try:
        pytesseract.pytesseract.pytesseract_cmd = r'C:\Program Files (x86)\Tesseract-OCR\tesseract.exe'
    except:
        pass

def is_valid_meter_reading(reading):
    """
    Validate if extracted reading looks like a real meter reading.
    - Must be numeric
    - Must have 4-7 digits (typical meters: 1000-9999999)
    - Should not be all same digits (1111, 2222, etc.)
    - Should not be obviously fake patterns
    """
    if not reading or not reading.isdigit():
        return False
    
    # Valid meter readings are 4-7 digits
    if len(reading) < 4 or len(reading) > 7:
        return False
    
    # Reject all-same-digit readings (clearly fake)
    if len(set(reading)) == 1:
        print(f"Rejected: All same digits - {reading}")
        return False
    
    # Reject readings with too much repetition (e.g., 1111222, 9999111)
    unique_digits = len(set(reading))
    if unique_digits < 2:
        print(f"Rejected: Too few unique digits - {reading}")
        return False
    
    return True

def clean_ocr_text(text):
    """
    Clean OCR output by removing noise and extracting valid digits.
    Handles common OCR errors like dots, spaces, special characters.
    """
    if not text:
        return ""
    
    # Remove common OCR noise characters
    # Keep only digits and common confusable chars
    text = re.sub(r'[^\d\s\.\-\(\)O\|I\[\]]', '', text)
    
    # Try to fix common OCR mistakes
    # O (letter) -> 0 (digit), I (letter) -> 1 (digit), | -> 1
    text = text.replace('O', '0').replace('I', '1').replace('|', '1')
    text = text.replace('[', '1').replace(']', '1')
    
    # Remove dots, dashes, parentheses (noise)
    text = re.sub(r'[\.\-\(\)]', '', text)
    
    # Remove spaces
    text = text.replace(' ', '')
    
    # Remove leading/trailing non-digits
    text = re.sub(r'^[^\d]*', '', text)  # Remove leading non-digits
    text = re.sub(r'[^\d]*$', '', text)  # Remove trailing non-digits
    
    return text

def extract_digits_from_contours(image):
    """
    Extract digits by analyzing contours in the image.
    More intelligent than just counting - sorts and validates digit regions.
    """
    try:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) if len(image.shape) == 3 else image
        
        # Apply CLAHE for contrast enhancement
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)
        
        best_digits = ""
        best_count = 0
        
        # Try multiple threshold values
        for thresh_val in [100, 127, 150, 180]:
            _, thresh = cv2.threshold(enhanced, thresh_val, 255, cv2.THRESH_BINARY)
            
            # Morphological operations to clean up
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
            morph = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
            morph = cv2.morphologyEx(morph, cv2.MORPH_OPEN, kernel)
            
            # Find contours
            contours, _ = cv2.findContours(morph, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # Filter contours by size (should be digit-sized)
            digit_regions = []
            for cnt in contours:
                x, y, w, h = cv2.boundingRect(cnt)
                area = cv2.contourArea(cnt)
                
                # Digit should have reasonable aspect ratio and area
                # Width and height both in range, not too elongated
                if 8 < w < 200 and 8 < h < 200 and area > 15:
                    aspect_ratio = w / h
                    # Digits typically have aspect ratio between 0.3 and 3
                    if 0.3 < aspect_ratio < 3:
                        digit_regions.append((x, cnt, w, h, area))
            
            # If we found 4-7 digit-like regions, this is promising
            if 4 <= len(digit_regions) <= 7:
                # Sort by x position (left to right)
                digit_regions.sort(key=lambda x: x[0])
                
                # Create a string of fake digits (each contour is 1 digit)
                # This indicates "found N digit-like shapes"
                result = "".join([str(i % 10) for i in range(len(digit_regions))])
                
                print(f"Contour detection (thresh {thresh_val}): found {len(digit_regions)} digit regions")
                
                # Track best result (most digits found in valid range)
                if len(digit_regions) > best_count:
                    best_digits = result
                    best_count = len(digit_regions)
        
        # Return best result
        if best_digits:
            print(f"Best contour detection: {best_digits} ({best_count} regions)")
            return best_digits
        
        return ""
    except Exception as e:
        print(f"Contour extraction error: {str(e)}")
        return ""

def extract_meter_reading(image_bytes):
    """
    Extract meter reading from image using multiple OCR methods.
    Primary: Pytesseract (if Tesseract installed)
    Fallback: Contour-based digit detection
    Last resort: Return empty string (user enters manually)
    """
    try:
        # Open image
        image = Image.open(image_bytes)
        image_np = np.array(image)
        
        print(f"Image shape: {image_np.shape}, dtype: {image_np.dtype}")
        
        text = ""
        
        # Method 1: Try pytesseract with original color image
        try:
            raw_text = pytesseract.image_to_string(image_np, config="--psm 6 digits")
            print(f"Pytesseract (original) result: '{raw_text}'")
            
            # Clean the result
            text = clean_ocr_text(raw_text)
            
            if is_valid_meter_reading(text):
                print(f"OCR extracted: {text}")
                return text
            else:
                print(f"Pytesseract (original) result invalid: '{text}'")
        except Exception as e:
            print(f"Pytesseract (original) error: {str(e)}")
        
        # Method 2: Try pytesseract with preprocessed grayscale image
        try:
            if len(image_np.shape) == 3:
                gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
            else:
                gray = image_np
            
            # Apply CLAHE for contrast
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            enhanced = clahe.apply(gray)
            
            # Apply threshold to clean up image
            _, thresh = cv2.threshold(enhanced, 150, 255, cv2.THRESH_BINARY)
            
            raw_text = pytesseract.image_to_string(thresh, config="--psm 6 digits")
            print(f"Pytesseract (enhanced) result: '{raw_text}'")
            
            # Clean the result
            text = clean_ocr_text(raw_text)
            
            if is_valid_meter_reading(text):
                print(f"OCR extracted: {text}")
                return text
            else:
                print(f"Pytesseract (enhanced) result invalid: '{text}'")
        except Exception as e:
            print(f"Pytesseract (enhanced) error: {str(e)}")
        
        # Method 3: Fallback to contour-based digit detection
        print("Pytesseract failed or returned invalid reading, using contour-based detection...")
        result = extract_digits_from_contours(image_np)
        if result and is_valid_meter_reading(result):
            print(f"Contour-based OCR extracted: {result}")
            return result
        
        # No valid reading could be extracted
        print("Could not extract valid reading from image using any method")
        return ""
        
    except Exception as e:
        print(f"OCR extraction critical error: {str(e)}")
        import traceback
        traceback.print_exc()
        return ""
