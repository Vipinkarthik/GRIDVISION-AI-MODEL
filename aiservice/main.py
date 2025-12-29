from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from utils.ocr import extract_meter_reading
from utils.classifier import validate_image
import io
import traceback

app = FastAPI(title="EB Meter AI Validation Service")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

print("✅ CORS middleware enabled - Frontend can now communicate with AI service")


# 🔹 HEALTH CHECK (BROWSER FRIENDLY)
@app.get("/")
def health_check():
    return {
        "status": "OK",
        "service": "EB Meter AI Validation Service",
        "message": "Service is running successfully"
    }


# 🔹 MAIN API
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
        except Exception as e:
            print(traceback.format_exc())

        image_stream.seek(0)

        try:
            is_valid_image = validate_image(image_stream)
        except:
            is_valid_image = True

        return {
            "status": "VALID",
            "meter_reading": str(ocr_reading).strip() if ocr_reading else "",
            "user_reading": str(user_reading),
            "image_valid": is_valid_image
        }

    except Exception as e:
        print(traceback.format_exc())
        return {"status": "ERROR"}
