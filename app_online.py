import os
from pathlib import Path
import sys
import streamlit as st

# Set page config as the very first Streamlit command
st.set_page_config(page_title="Arabic ANPR Demo", layout="wide")

import cv2
import numpy as np
from PIL import Image

from ultralytics import YOLO
from paddleocr import PaddleOCR

# ------------------------
# Configuration Variables
# ------------------------

# Use relative paths for models for portability
CAR_MODEL_PATH = Path("yolov8n.pt")
PLATE_MODEL_PATH = Path("vehicle_number_plate_detector.pt")
VEHICLE_CLASSES = [2, 3, 5, 7]  # COCO: car, motorcycle, bus, truck

# ------------------------
# Utility Functions
# ------------------------

@st.cache_resource
def load_models():
    """Load YOLO and OCR models, cached for performance."""
    st.info("Loading models (YOLO and OCR)...")
    try:
        if not CAR_MODEL_PATH.exists():
            st.error(f"Car detection model not found at {CAR_MODEL_PATH.resolve()}")
            raise FileNotFoundError(f"Car detection model not found at {CAR_MODEL_PATH.resolve()}")
        if not PLATE_MODEL_PATH.exists():
            st.error(f"Plate detection model not found at {PLATE_MODEL_PATH.resolve()}")
            raise FileNotFoundError(f"Plate detection model not found at {PLATE_MODEL_PATH.resolve()}")

        car_model = YOLO(str(CAR_MODEL_PATH))
        car_model.to('cpu')
        plate_model = YOLO(str(PLATE_MODEL_PATH))
        plate_model.to('cpu')
        ocr_en = PaddleOCR(lang='en', use_textline_orientation=True)
        ocr_ar = PaddleOCR(lang='ar', use_textline_orientation=True)
        st.info("Models loaded successfully.")
        return car_model, plate_model, ocr_en, ocr_ar
    except Exception as e:
        st.error(f"Error loading models: {e}")
        raise

def to_cv2_image(uploaded_file):
    """Convert uploaded file to OpenCV image."""
    try:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        return img
    except Exception as e:
        st.error(f"Failed to decode image: {e}")
        return None

def detect_vehicle_boxes(car_model, image):
    """Detect vehicles in the image using YOLO."""
    try:
        results = car_model.predict(image, conf=0.3, verbose=False)[0]
        boxes = []
        for box in results.boxes:
            cls_id = int(box.cls)
            if cls_id in VEHICLE_CLASSES:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                boxes.append((x1, y1, x2, y2))
        return boxes
    except Exception as e:
        st.error(f"Vehicle detection failed: {e}")
        return []

def detect_plate_boxes(plate_model, vehicle_crop):
    """Detect plates in the cropped vehicle image."""
    try:
        results = plate_model.predict(vehicle_crop, conf=0.6, verbose=False)[0]
        boxes = []
        for box in results.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            boxes.append((x1, y1, x2, y2))
        return boxes
    except Exception as e:
        st.error(f"Plate detection failed: {e}")
        return []

def extract_plate_text(ocr_en, ocr_ar, plate_crop, ocr_lang):
    """Extract text from the cropped plate image using OCR in the selected language."""
    try:
        plate_rgb = cv2.cvtColor(plate_crop, cv2.COLOR_BGR2RGB)
        texts = []
        def extract_from_ocr_result(ocr_result):
            # ocr_result: [ [ [box, (text, score)], ... ] ]
            out = []
            for line in ocr_result:
                for item in line:
                    text, score = item[1]
                    if score > 0.3 and text.strip():
                        out.append(text)
            return out

        if ocr_lang == "English":
            results = ocr_en.ocr(plate_rgb)
            texts.extend(extract_from_ocr_result(results))
        elif ocr_lang == "Arabic":
            results = ocr_ar.ocr(plate_rgb)
            texts.extend(extract_from_ocr_result(results))
        else:  # Both
            results_en = ocr_en.ocr(plate_rgb)
            results_ar = ocr_ar.ocr(plate_rgb)
            texts.extend(extract_from_ocr_result(results_en))
            for t in extract_from_ocr_result(results_ar):
                if t not in texts:
                    texts.append(t)
        return " ".join(texts).strip()
    except Exception as e:
        st.error(f"OCR failed: {e}")
        return ""

def draw_boxes_and_labels(image, vehicle_boxes, plate_info):
    """Draw bounding boxes for vehicles (green) and plates (blue), and overlay text."""
    for (x1, y1, x2, y2) in vehicle_boxes:
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 3)
    for info in plate_info:
        (px1, py1, px2, py2, text, offset_x, offset_y) = info
        cv2.rectangle(image, (offset_x + px1, offset_y + py1), (offset_x + px2, offset_y + py2), (255, 0, 0), 3)
        if text:
            cv2.putText(
                image,
                text,
                (offset_x + px1, offset_y + py1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 0, 0),
                2,
                cv2.LINE_AA,
            )
    return image

# ------------------------
# Streamlit App Layout
# ------------------------

# --- Custom CSS for beautiful UI ---
st.markdown("""
    <style>
    body {
        background: linear-gradient(135deg, #f8fafc 0%, #e0e7ff 100%);
    }
    .main {
        background: transparent !important;
    }
    .stApp {
        font-family: 'Segoe UI', 'Roboto', 'Arial', sans-serif;
        color: #22223b;
    }
    .anpr-header {
        background: linear-gradient(90deg, #4f8cff 0%, #38b6ff 100%);
        color: white;
        border-radius: 18px;
        padding: 1.5rem 2rem 1rem 2rem;
        margin-bottom: 2rem;
        box-shadow: 0 4px 24px 0 rgba(80, 120, 255, 0.10);
        text-align: center;
    }
    .anpr-card {
        background: #fff;
        border-radius: 16px;
        box-shadow: 0 2px 16px 0 rgba(80, 120, 255, 0.08);
        padding: 1.5rem 1.5rem 1rem 1.5rem;
        margin-bottom: 1.5rem;
    }
    .anpr-divider {
        border-top: 2px solid #4f8cff;
        margin: 1.5rem 0 1.5rem 0;
    }
    .anpr-plate {
        font-size: 1.2rem;
        font-weight: 600;
        color: #1b263b;
        background: #e0e7ff;
        border-radius: 8px;
        padding: 0.5rem 1rem;
        display: inline-block;
        margin-bottom: 0.5rem;
    }
    </style>
""", unsafe_allow_html=True)

# --- Header with icon and title ---
st.markdown("""
    <div class="anpr-header">
        <h1 style="margin-bottom:0.2em;">🚗🔎 Arabic ANPR Streamlit App</h1>
        <p style="font-size:1.1em; margin-bottom:0;">Detect vehicles, license plates, and extract Arabic plate!</p>
    </div>
""", unsafe_allow_html=True)

st.markdown('<span style="color:white; font-size:1.2em;">📤 Upload an image to get started</span>', unsafe_allow_html=True)
uploaded_file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])

# Language selection for OCR
ocr_lang = st.selectbox(
    "Select OCR Language",
    ["English", "Arabic", "Both"],
    index=2,
    help="Choose the language(s) for number plate text extraction."
)

process_clicked = False
if uploaded_file:
    print(f"[LOG] Image uploaded: {uploaded_file.name}")
    process_clicked = st.button("Process", type="primary")
    if process_clicked:
        st.info("Processing started...")
        print("[LOG] Processing started...")

        print("[LOG] Loading models...")
        car_model, plate_model, ocr_en, ocr_ar = load_models()
        print("[LOG] Models loaded.")

        st.info("Converting uploaded image to OpenCV format...")
        print("[LOG] Converting uploaded image to OpenCV format...")
        image = to_cv2_image(uploaded_file)
        if image is not None:
            orig_image = image.copy()
            st.info("Detecting vehicles in the image...")
            print("[LOG] Performing vehicle detection...")
            vehicle_boxes = detect_vehicle_boxes(car_model, image)
            st.info(f"Detected {len(vehicle_boxes)} vehicles.")
            print(f"[LOG] Detected {len(vehicle_boxes)} vehicles.")

            plate_info = []
            for (x1, y1, x2, y2) in vehicle_boxes:
                print(f"[LOG] Cropping vehicle region: ({x1}, {y1}, {x2}, {y2})")
                vehicle_crop = orig_image[y1:y2, x1:x2]
                st.info("Detecting plates in vehicle region...")
                print("[LOG] Performing plate detection in vehicle region...")
                plate_boxes = detect_plate_boxes(plate_model, vehicle_crop)
                st.info(f"Detected {len(plate_boxes)} plates in vehicle.")
                print(f"[LOG] Detected {len(plate_boxes)} plates in vehicle.")

                for (px1, py1, px2, py2) in plate_boxes:
                    print(f"[LOG] Cropping plate region: ({px1}, {py1}, {px2}, {py2})")
                    plate_crop = vehicle_crop[py1:py2, px1:px2]
                    if plate_crop.size == 0:
                        print("[LOG] Plate crop is empty, skipping.")
                        continue
                    st.info("Running OCR on detected plate...")
                    print("[LOG] Performing OCR on cropped plate...")
                    text = extract_plate_text(ocr_en, ocr_ar, plate_crop, ocr_lang)
                    st.info(f"OCR result: {text}")
                    print(f"[LOG] OCR result: {text}")
                    plate_info.append((px1, py1, px2, py2, text, x1, y1))

            st.info("Drawing boxes and labels on image...")
            print("[LOG] Drawing boxes and labels on image...")
            annotated = draw_boxes_and_labels(orig_image, vehicle_boxes, plate_info)
            annotated_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)

            # --- Use columns for side-by-side layout ---
            col1, col2 = st.columns([1, 1.2])
            with col1:
                st.markdown('<span style="color:white; font-size:1.2em;">🖼️ Uploaded Image</span>', unsafe_allow_html=True)
                st.image(uploaded_file, use_container_width=True)
            with col2:
                st.markdown('<span style="color:white; font-size:1.2em;">🎯 Detection Results</span>', unsafe_allow_html=True)
                st.image(annotated_rgb, caption="Detected Vehicles (green) & Plates (blue)", use_container_width=True)

            st.markdown('<div class="anpr-divider"></div>', unsafe_allow_html=True)

            # --- Show cropped plate images and extracted text in a modern, multilingual card UI ---
            if plate_info:
                st.markdown("""
                        <h4 style="color:#38b6ff; margin-bottom:0.5em;">🔠 Extracted Plate(s)</h4>
                        <div style="display: flex; flex-wrap: wrap; gap: 2em;">
                """, unsafe_allow_html=True)
                for idx, info in enumerate(plate_info, start=1):
                    px1, py1, px2, py2, text, x1, y1 = info
                    # Crop the plate image for display
                    plate_crop = orig_image[y1:y2, x1:x2][py1:py2, px1:px2]
                    if plate_crop.size == 0:
                        continue
                    # Convert to RGB for Streamlit
                    plate_crop_rgb = cv2.cvtColor(plate_crop, cv2.COLOR_BGR2RGB)
                    # Determine text direction for Arabic/English
                    is_arabic = any('\u0600' <= c <= '\u06FF' for c in text)
                    direction = "rtl" if is_arabic else "ltr"
                    # Card for each plate: just the cropped image and centered white text
                    st.image(plate_crop_rgb, width=180, caption="Plate Crop", use_container_width=False)
                    st.markdown(
                        f"""
                        <div style="text-align:center; margin-top:0.5em; margin-bottom:1.5em;">
                            <span style="font-size:1.3em; font-weight:700; color:#fff; display:inline-block; direction:{direction}; unicode-bidi:plaintext; font-family:'Noto Sans Arabic','Cairo','Amiri','Segoe UI','Arial',sans-serif;">
                                {text if text else '<span style="color:#aaa;">No text detected</span>'}
                            </span>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
                st.markdown("</div></div>", unsafe_allow_html=True)
                print(f"[LOG] Final OCR results: {[info[4] for info in plate_info]}")
            else:
                st.info("No plate text extracted.")
                print("[LOG] No plate text extracted.")
        else:
            st.error("Failed to load image. Please try another file.")
            print("[LOG] Failed to load image. Please try another file.")
        st.markdown('</div>', unsafe_allow_html=True)
