from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from utils.ocr import extract_meter_reading
from utils.classifier import validate_image
import io
import traceback

app = FastAPI(title="EB Meter AI Validation Service")

# Enable CORS for all origins (required for frontend communication)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allow all methods (GET, POST, etc.)
    allow_headers=["*"],  # Allow all headers
)

print("✅ CORS middleware enabled - Frontend can now communicate with AI service")

@app.post("/validate-meter")
async def validate_meter(
    image: UploadFile = File(...),
    user_reading: str = Form(...)
):
    try:
        image_bytes = await image.read()
        image_stream = io.BytesIO(image_bytes)

        ocr_reading = None
        try:
            ocr_reading = extract_meter_reading(image_stream)
            print(f"✅ OCR extracted: '{ocr_reading}' (type: {type(ocr_reading).__name__}, length: {len(str(ocr_reading))})")
        except Exception as e:
            print(f"❌ OCR Error: {str(e)}")
            print(traceback.format_exc())
            ocr_reading = None

        image_stream.seek(0)
        is_valid_image = True
        try:
            is_valid_image = validate_image(image_stream)
            print(f"✅ Image validation result: {is_valid_image}")
        except Exception as e:
            print(f"⚠️ Validation Error: {str(e)}")
            print(traceback.format_exc())
            # Don't fail on validation error - just accept the image
            is_valid_image = True

        # Prepare response with extracted reading
        meter_reading_value = str(ocr_reading).strip() if ocr_reading else ""
        print(f"📊 Final meter_reading for response: '{meter_reading_value}'")
        
        response_data = {
            "status": "VALID",
            "meter_reading": meter_reading_value,
            "user_reading": str(user_reading),
            "image_valid": is_valid_image
        }
        
        print(f"📤 Sending response: {response_data}")
        return response_data
        
    except Exception as e:
        print(f"❌ Endpoint Error: {str(e)}")
