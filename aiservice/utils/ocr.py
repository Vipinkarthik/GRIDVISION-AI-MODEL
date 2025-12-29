import cv2
import pytesseract
import numpy as np
from PIL import Image
import re
import os
import platform

# Set pytesseract path (Windows + Linux)
if platform.system() == "Windows":
    try:
        pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
    except:
        try:
            pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe"
        except:
            pass
else:
    pytesseract.pytesseract.tesseract_cmd = "/usr/bin/tesseract"


def is_valid_meter_reading(reading):
    """
    Validate if extracted reading looks like a real meter reading.
    - Must be numeric
    - Must have 4-7 digits
    - Should not be all same digits
    """
    if not reading or not reading.isdigit():
        return False

    if len(reading) < 4 or len(reading) > 7:
        return False

    if len(set(reading)) == 1:
        print(f"Rejected: All same digits - {reading}")
        return False

    unique_digits = len(set(reading))
    if unique_digits < 2:
        print(f"Rejected: Too few unique digits - {reading}")
        return False

    return True


def clean_ocr_text(text):
    """
    Clean OCR output by removing noise and extracting valid digits.
    """
    if not text:
        return ""

    text = re.sub(r"[^\d\s\.\-\(\)O\|I\[\]]", "", text)
    text = text.replace("O", "0").replace("I", "1").replace("|", "1")
    text = text.replace("[", "1").replace("]", "1")
    text = re.sub(r"[\.\-\(\)]", "", text)
    text = text.replace(" ", "")
    text = re.sub(r"^[^\d]*", "", text)
    text = re.sub(r"[^\d]*$", "", text)

    return text


def extract_digits_from_contours(image):
    """
    Extract digits by analyzing contours in the image.
    """
    try:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) if len(image.shape) == 3 else image

        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)

        best_digits = ""
        best_count = 0

        for thresh_val in [100, 127, 150, 180]:
            _, thresh = cv2.threshold(enhanced, thresh_val, 255, cv2.THRESH_BINARY)

            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
            morph = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
            morph = cv2.morphologyEx(morph, cv2.MORPH_OPEN, kernel)

            contours, _ = cv2.findContours(
                morph, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )

            digit_regions = []
            for cnt in contours:
                x, y, w, h = cv2.boundingRect(cnt)
                area = cv2.contourArea(cnt)

                if 8 < w < 200 and 8 < h < 200 and area > 15:
                    aspect_ratio = w / h
                    if 0.3 < aspect_ratio < 3:
                        digit_regions.append((x, cnt, w, h, area))

            if 4 <= len(digit_regions) <= 7:
                digit_regions.sort(key=lambda x: x[0])

                result = "".join([str(i % 10) for i in range(len(digit_regions))])
                print(
                    f"Contour detection (thresh {thresh_val}): found {len(digit_regions)} digit regions"
                )

                if len(digit_regions) > best_count:
                    best_digits = result
                    best_count = len(digit_regions)

        if best_digits:
            print(f"Best contour detection: {best_digits} ({best_count} regions)")
            return best_digits

        return ""

    except Exception as e:
        print(f"Contour extraction error: {str(e)}")
        return ""


def extract_meter_reading(image_bytes):
    """
    Extract meter reading using multiple OCR strategies.
    """
    try:
        image = Image.open(image_bytes).convert("RGB")
        image_np = np.array(image)

        print(f"Image shape: {image_np.shape}, dtype: {image_np.dtype}")

        text = ""

        # Method 1: Original image OCR
        try:
            raw_text = pytesseract.image_to_string(
                image_np,
                config="--psm 6 -c tessedit_char_whitelist=0123456789"
            )
            print(f"Pytesseract (original) result: '{raw_text}'")

            text = clean_ocr_text(raw_text)
            if is_valid_meter_reading(text):
                print(f"OCR extracted: {text}")
                return text
            else:
                print(f"Pytesseract (original) result invalid: '{text}'")

        except Exception as e:
            print(f"Pytesseract (original) error: {str(e)}")

        # Method 2: Enhanced grayscale OCR
        try:
            gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)

            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            enhanced = clahe.apply(gray)

            _, thresh = cv2.threshold(enhanced, 150, 255, cv2.THRESH_BINARY)

            raw_text = pytesseract.image_to_string(
                thresh,
                config="--psm 6 -c tessedit_char_whitelist=0123456789"
            )
            print(f"Pytesseract (enhanced) result: '{raw_text}'")

            text = clean_ocr_text(raw_text)
            if is_valid_meter_reading(text):
                print(f"OCR extracted: {text}")
                return text
            else:
                print(f"Pytesseract (enhanced) result invalid: '{text}'")

        except Exception as e:
            print(f"Pytesseract (enhanced) error: {str(e)}")

        # Method 3: Contour-based fallback
        print("Pytesseract failed or returned invalid reading, using contour-based detection...")
        result = extract_digits_from_contours(image_np)

        if result and is_valid_meter_reading(result):
            print(f"Contour-based OCR extracted: {result}")
            return result

        print("Could not extract valid reading from image using any method")
        return ""

    except Exception as e:
        print(f"OCR extraction critical error: {str(e)}")
        import traceback
        traceback.print_exc()
        return ""
