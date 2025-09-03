# Viejo Garage - Smart Parking Space Detection

AI-powered parking garage monitoring system that accurately detects occupied/available spaces even through garage gates and with partial vehicle occlusion.

## ✨ Features

- 🚗 **Accurate vehicle detection** using enhanced YOLOv8 with multiple preprocessing techniques
- 🎯 **Handles occlusion** - detects vehicles behind gates and bars
- 🎨 **Color-based validation** - secondary detection using car-like colors
- 📐 **Custom space calibration** - draw parking spaces for any garage layout
- 💯 **High accuracy** - multiple detection methods with voting system
- 🚀 **Fast processing** - under 2 seconds per image
- 💾 **100% local** - no cloud dependencies

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
cd /Users/emilianoperez/Projects/01-wyeworks/viejo-garage

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Basic Usage

```bash
# Process a single image
python main.py path/to/garage-image.jpg

# Process all images in test_images/
python main.py

# Calibrate parking spaces for a new image
python calibrate.py path/to/garage-image.jpg
```

## 🎯 How It Works

### Enhanced Detection Pipeline

The system uses multiple image processing techniques to detect vehicles even when partially obscured:

1. **Multi-Processing**: Each image is processed 6 different ways:
   - Original
   - CLAHE contrast enhancement
   - Brightness adjustment
   - Edge enhancement (best for obscured vehicles)
   - Denoising
   - Gate line removal

2. **YOLOv8 Detection**: Runs on each processed version with low confidence threshold (0.1)

3. **Color Detection**: Validates by checking for car-like colors in each space

4. **Voting System**: Combines results - if multiple methods agree, confidence increases

### Calibration Mode

Draw custom parking spaces for any garage:

```bash
python calibrate.py test_images/your-garage.jpg
```

- Click and drag to draw rectangles around parking spaces
- Press 'r' to remove last space
- Press 'c' to clear all
- Press 's' to save configuration
- Press 'q' or ESC to quit

Configurations are saved per image and automatically loaded.

## 📊 Performance

- **Detection accuracy**: 100% on test images (both vehicles detected)
- **Processing time**: ~1.5 seconds per image
- **Consistency**: Deterministic results (not random like LLMs)

## 🏗️ Project Structure

```
viejo-garage/
├── main.py                  # Main detection script
├── enhanced_detector.py     # Enhanced multi-method detection
├── detector.py             # Basic YOLO detection
├── parking_spaces.py       # Parking space management
├── calibrate.py           # Interactive calibration tool
├── config.py              # Configuration settings
├── requirements.txt       # Python dependencies
├── test_images/          # Test images directory
├── parking_configs/      # Saved parking space configurations
└── output/              # Detection results
```

## 🔧 Configuration

Edit `config.py` to adjust:

- `YOLO_MODEL`: YOLOv8 model variant (default: yolov8l.pt)
- `YOLO_CONFIDENCE`: Detection threshold (default: 0.1)
- `PARKING_SPACE_IOU_THRESHOLD`: Overlap threshold (default: 0.3)

## 🎬 Next Steps

Ready for:
- [ ] Video stream processing
- [ ] Webhook notifications (Slack integration)
- [ ] Real-time monitoring dashboard
- [ ] Multiple camera support
- [ ] Historical tracking and analytics

## 📝 Example Output

```
============================================================
Processing: PHOTO-2025-09-03-15-52-58.jpg
============================================================
Image dimensions: 1600x1200

Using Enhanced Detection with pre-processing...
Processing 2 parking spaces...

Parking Status:
  Total spaces: 2
  Occupied: 2
  Available: 0
  Occupancy rate: 100.0%
  Occupied spaces: P1, P2
============================================================
```

## 🛠️ Troubleshooting

### Vehicles not detected?
1. Run calibration to ensure parking spaces are correctly defined
2. Check image lighting - very dark images may need adjustment
3. Ensure gates/obstacles aren't completely blocking vehicles

### Best practices for images:
- Good lighting (daylight or well-lit garage)
- Clear view of parking spaces
- Minimal obstruction of vehicles
- Image resolution: 1280x720 or higher

## 📄 License

MIT