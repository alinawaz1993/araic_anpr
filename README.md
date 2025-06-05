# UAE License Plate Recognition (ANPR) Pipeline

This project implements a two-stage ANPR pipeline for UAE license plates using YOLOv11 for detection and PaddleOCR for character recognition.

## Project Structure

```
uae_anpr/
├── detection/                         # YOLOv11 detection training
│   ├── data/
│   │   ├── images/
│   │   │   ├── train/                # 80% of images
│   │   │   └── val/                  # 20% of images
│   │   └── labels/
│   │       ├── train/                # Training labels
│   │       └── val/                  # Validation labels
│   ├── configs/
│   │   └── yolov11_uae_car.yaml      # YOLOv11 config
│   ├── weights/                      # Model weights
│   └── scripts/                      # Helper scripts
│
├── recognition/                       # OCR training
│   ├── data/                         # Cropped images
│   ├── labels/                       # OCR labels
│   ├── configs/
│   │   └── rec_uae_plate.yaml        # OCR config
│   ├── dicts/
│   │   └── uae_plate_dict.txt        # Character dictionary
│   └── weights/                      # OCR weights
│
└── inference/                        # Inference pipeline
    ├── detection_model/              # Exported detection model
    ├── ocr_model/                    # Exported OCR model
    ├── scripts/
    │   └── anpr_pipeline.py          # Main pipeline script
    └── outputs/                      # Results and visualizations
```

## Dataset

The project uses the "uae car" dataset containing 8,190 images with 53 classes:
- Numeric digits (0-9)
- Alphabet letters (A-Z)
- Vehicle and location labels (plate, back_car, front_car, etc.)

## Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Prepare the dataset:
```bash
python detection/scripts/prepare_data.py
```

3. Train the detection model:
```bash
cd detection
yolo train --config configs/yolov11_uae_car.yaml
```

4. Train the OCR model:
```bash
cd recognition
python tools/train.py -c configs/rec_uae_plate.yaml
```

## Usage

Run inference on new images:
```bash
python inference/scripts/anpr_pipeline.py
```

The pipeline will:
1. Detect license plates and characters using YOLOv11
2. Group characters by plate
3. Recognize characters using PaddleOCR
4. Save results and visualizations

## Output

The pipeline generates:
- JSON files with detection results
- Visualized images with bounding boxes and recognized text
- Log files with detailed information

## License

This project is licensed under the MIT License - see the LICENSE file for details. 