# Emotion Detection System

A real-time emotion detection system that uses your webcam and OpenCV to detect and analyze emotions.

![Emotion Detection Demo](demo.gif)

## Features

- Real-time video feed from webcam
- Face detection using Haar Cascade
- Emotion detection based on facial features
- Color-coded emotion display
- Simple and lightweight

## Setup Instructions

### 1. Prerequisites

- Python 3.8+
- Webcam

### 2. Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/emotion-detection.git
cd emotion-detection
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

### 3. Running the Application

Start the application:
```bash
python emotion_detection.py
```

## Usage

1. Make sure you have a working webcam
2. Run the application
3. Look at the webcam and try different facial expressions
4. The application will detect your face and show the detected emotion
5. Press 'q' to quit the application

### Emotion Colors

- 😊 Happy: Green
- 😢 Sad: Blue
- 😠 Angry: Red
- 😲 Surprised: Cyan
- 😐 Neutral: White
- 😴 Tired: Gray
- 😨 Fearful: Magenta
- 🤢 Disgusted: Dark Green

## Requirements

- OpenCV
- NumPy
- Python 3.8+

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Acknowledgments

- OpenCV for the face detection models
- Python for making it all possible
