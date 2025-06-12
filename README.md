# 🦅 R.A.P.T.O.R
**Real-time Aerial Patrol and Tactical Object Recognition**

Advanced AI-powered object detection system for tactical surveillance and reconnaissance operations.

## 🎯 Overview

R.A.P.T.O.R is a cutting-edge tactical detection system that combines YOLO object detection with GPS mapping capabilities to provide real-time intelligence from aerial footage. Perfect for military, law enforcement, search and rescue, and security applications.

### Key Features
- **Real-time Detection**: YOLO-based object recognition at >15 FPS
- **GPS Mapping**: Automatic conversion of detections to GPS coordinates  
- **Multi-format Output**: JSON, CSV, GeoJSON for integration with other systems
- **Tactical Focus**: Optimized for vehicles, personnel, and equipment detection
- **Modular Design**: Easy to extend and customize for specific missions

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- Git
- VS Code (recommended)

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/happyylad/R.A.P.T.O.R.git
cd R.A.P.T.O.R
```

2. **Set up virtual environment**
```bash
python -m venv .venv
# Windows
.\.venv\Scripts\Activate.ps1
# Mac/Linux  
source .venv/bin/activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Run R.A.P.T.O.R**
```bash
python src/main.py
```

## 📋 Project Structure

```
R.A.P.T.O.R/
├── src/                    # Source code
│   ├── main.py            # Main R.A.P.T.O.R system
│   └── ...                # Additional modules
├── tests/                  # Test files
├── docs/                   # Documentation
├── output/                 # Detection results (auto-created)
│   ├── detections/        # JSON detection files
│   ├── images/            # Annotated images
│   ├── videos/            # Processed videos
│   └── maps/              # Generated maps
├── requirements.txt        # Python dependencies
└── README.md              # This file
```

## 🎮 Usage

### Basic Image Detection
```python
from src.main import RaptorDetectionSystem

# Initialize R.A.P.T.O.R
raptor = RaptorDetectionSystem()

# Process image
detections = raptor.process_image("path/to/image.jpg")

# Generate report
raptor.generate_report()
```

### Video Processing
```python
# Process video with GPS mapping
gps_bounds = {
    'top_left': {'lat': 36.4074, 'lon': -105.5731},
    'top_right': {'lat': 36.4074, 'lon': -105.5700},
    'bottom_left': {'lat': 36.4044, 'lon': -105.5731},
    'bottom_right': {'lat': 36.4044, 'lon': -105.5700}
}

raptor = RaptorDetectionSystem(gps_bounds=gps_bounds)
detections = raptor.process_video("drone_footage.mp4")
```

## 🔧 Configuration

### Supported Object Classes
- **Personnel**: Person detection and tracking
- **Vehicles**: Cars, trucks, buses, motorcycles
- **Wildlife**: Birds, animals (for environmental monitoring)

### GPS Mapping
Configure GPS bounds for your area of operations in the `gps_bounds` parameter.

## 📊 Output Formats

R.A.P.T.O.R generates multiple output formats:
- **JSON**: Detailed detection data with coordinates
- **Annotated Images/Videos**: Visual results with bounding boxes
- **GeoJSON**: GPS-enabled mapping data
- **Tactical Reports**: Summary statistics and analysis

## 🤝 Collaboration

### For Team Members

1. **Clone the project**
```bash
git clone https://github.com/happyylad/R.A.P.T.O.R.git
cd R.A.P.T.O.R
```

2. **Configure Git with YOUR information**
```bash
git config --global user.name "Your Name"
git config --global user.email "your-email@gmail.com"
```

3. **Set up environment and start coding**
```bash
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
code .
```

4. **Daily workflow**
```bash
git pull                    # Get latest changes
# ... make your changes ...
git add .
git commit -m "Description of changes"
git push                    # Share your work
```

## 🛠️ Development

### Running Tests
```bash
pytest tests/
```

### Code Formatting
```bash
black src/
```

### Adding New Features
1. Create feature branch: `git checkout -b feature-name`
2. Implement changes in `src/`
3. Add tests in `tests/`
4. Update documentation
5. Submit pull request

## 📈 Performance

- **Processing Speed**: 15-30 FPS on standard hardware
- **Detection Accuracy**: 80-95% confidence on tactical objects
- **GPS Precision**: <10 meter accuracy
- **Real-time Capable**: Yes, with frame skipping optimization

## 🎯 Use Cases

### Military Applications
- Base perimeter security
- Convoy route surveillance  
- Intelligence gathering
- Force protection

### Civilian Applications
- Search and rescue operations
- Border security monitoring
- Critical infrastructure protection
- Emergency response coordination

## 📚 Documentation

- [Installation Guide](docs/INSTALLATION.md)
- [User Manual](docs/USER_GUIDE.md)
- [API Reference](docs/API_REFERENCE.md)
- [Development Guide](docs/DEVELOPMENT.md)

## 🐛 Troubleshooting

### Common Issues

**YOLO model not found**
```bash
pip install ultralytics
```

**OpenCV errors**
```bash
pip install opencv-python-headless
```

**Permission errors on Windows**
```bash
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **YOLO**: Ultralytics for the detection model
- **OpenCV**: Computer vision processing
- **Contributors**: All team members working on R.A.P.T.O.R

## 📞 Contact

For questions, issues, or contributions:
- **GitHub Issues**: [Report bugs or request features](https://github.com/happyylad/R.A.P.T.O.R/issues)
- **Team Lead**: [Your contact information]

---

**R.A.P.T.O.R - Providing superior tactical intelligence through advanced AI detection** 🦅