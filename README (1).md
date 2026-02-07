# VisionSpec QC - PCB Defect Detection System

A deep learning-based automated quality control system for detecting defects in Printed Circuit Boards (PCBs) using transfer learning with MobileNetV2.

## 🎯 Project Overview

This project implements an AI-powered visual inspection system that can automatically classify PCB images as either **PASS** (defect-free) or **DEFECT** (containing defects). The system uses transfer learning with MobileNetV2 as the base model, fine-tuned for PCB defect detection.

## ✨ Features

- **Image Pipeline & Preprocessing**: Comprehensive data loading and preprocessing pipeline
- **Data Augmentation**: Advanced augmentation techniques to improve model robustness
  - Rotation (±20°)
  - Width/Height shift (20%)
  - Brightness adjustment (80-120%)
  - Zoom (±15%)
  - Horizontal & Vertical flips
- **Transfer Learning**: MobileNetV2 pre-trained on ImageNet
- **Binary Classification**: Distinguishes between defective and non-defective PCBs
- **Visualization Tools**: Built-in functions to visualize augmentations and training batches

## 🏗️ Architecture

```
Input (224x224x3)
    ↓
MobileNetV2 (Frozen)
    ↓
Global Average Pooling
    ↓
Dense (256) + ReLU + Dropout(0.5)
    ↓
Dense (128) + ReLU + Dropout(0.5)
    ↓
Dense (1) + Sigmoid
    ↓
Output (PASS/DEFECT)
```

**Model Statistics:**
- Total parameters: 2,618,945
- Trainable parameters: 360,961 (1.38 MB)
- Non-trainable parameters: 2,257,984 (8.61 MB)

## 📋 Requirements

```
tensorflow>=2.13.0
numpy>=1.24.0
opencv-python>=4.8.0
matplotlib>=3.7.0
pillow>=10.0.0
```

## 🚀 Installation

1. **Clone the repository**
```bash
git clone https://github.com/yourusername/pcb-defect-detection.git
cd pcb-defect-detection
```

2. **Create a virtual environment** (recommended)
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

## 📁 Dataset Structure

Organize your PCB images in the following structure:

```
PCBData/
├── pass/           # Defect-free PCB images
│   ├── image1.jpg
│   ├── image2.jpg
│   └── ...
└── defect/         # Defective PCB images
    ├── image1.jpg
    ├── image2.jpg
    └── ...
```

## 💻 Usage

### Running the Jupyter Notebook

1. Start Jupyter Notebook:
```bash
jupyter notebook
```

2. Open `pcb_defect_detection.ipynb` and run the cells sequentially

### Training the Model

The notebook is organized into sections:

1. **Image Pipeline & Preprocessing**: Load and prepare data
2. **Transfer Learning Model Training**: Build and train the model
3. **Model Evaluation**: Evaluate performance on test data
4. **Inference**: Make predictions on new images

### Quick Start Example

```python
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Initialize pipeline
pipeline = ImagePipeline(img_size=(224, 224), batch_size=32)

# Create data generators
train_gen, val_gen = pipeline.create_dataset_from_directory('PCBData')

# Build model
model_builder = PCBModelBuilder(input_shape=(224, 224, 3))
model = model_builder.build_model()

# Compile and train
model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy', 'precision', 'recall', 'auc']
)

history = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=20
)
```

## 📊 Model Performance

The model is trained with the following configuration:
- **Optimizer**: Adam (learning rate: 0.001)
- **Loss Function**: Binary Crossentropy
- **Metrics**: Accuracy, Precision, Recall, AUC
- **Batch Size**: 32
- **Image Size**: 224×224×3

## 🔍 Key Components

### ImagePipeline Class
- Handles image loading and preprocessing
- Creates augmentation generators
- Generates train/validation datasets

### PCBModelBuilder Class
- Builds transfer learning model
- Configurable architecture
- Supports different base models

### Visualization Functions
- `visualize_batch()`: Display training batch samples
- `show_augmentations()`: Demonstrate augmentation effects

## 📝 Project Structure

```
pcb-defect-detection/
│
├── pcb_defect_detection.ipynb  # Main Jupyter notebook
├── README.md                     # Project documentation
├── requirements.txt              # Python dependencies
├── .gitignore                    # Git ignore file
├── LICENSE                       # License file
│
├── models/                       # Saved models (not in repo)
│   └── .gitkeep
│
└── data/                         # Dataset (not in repo)
    └── .gitkeep
```

## 🎓 Learning Objectives

This project demonstrates:
- Transfer learning with pre-trained CNNs
- Image data preprocessing and augmentation
- Binary classification for quality control
- Building production-ready ML pipelines
- Computer vision for industrial applications

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 👥 Authors

- Your Name - *Initial work*

## 🙏 Acknowledgments

- MobileNetV2 architecture by Google
- TensorFlow and Keras teams
- PCB dataset contributors

## 📧 Contact

For questions or feedback, please reach out to:
- Email: your.email@example.com
- GitHub: [@yourusername](https://github.com/yourusername)

## 🔮 Future Enhancements

- [ ] Multi-class defect classification (different defect types)
- [ ] Real-time inference with webcam
- [ ] Web application interface
- [ ] Model deployment using TensorFlow Serving
- [ ] Support for additional architectures (EfficientNet, ResNet)
- [ ] Explainability features (Grad-CAM visualizations)
- [ ] REST API for inference

---

**Note**: This project is for educational and research purposes in quality control and computer vision applications.
