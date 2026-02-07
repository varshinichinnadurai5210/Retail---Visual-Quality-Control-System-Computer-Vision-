# Trained Models Directory

This directory stores your trained model files.

## Saving Models

The training notebook will save models here in different formats:

### Keras/TensorFlow Format (.keras or .h5)

```python
# Save entire model
model.save('models/pcb_detector_v1.keras')

# Save weights only
model.save_weights('models/pcb_weights_v1.h5')
```

### Model Naming Convention

Use descriptive names with version numbers:

```
models/
├── pcb_mobilenetv2_v1.keras        # Base model
├── pcb_mobilenetv2_v2.keras        # Improved version
├── pcb_resnet50_v1.keras           # Different architecture
├── best_model.keras                # Best performing model
└── checkpoints/                     # Training checkpoints
    ├── epoch_10.keras
    ├── epoch_20.keras
    └── ...
```

## Model Information

Document your models in this table:

| Model Name | Architecture | Accuracy | Precision | Recall | Date | Notes |
|------------|-------------|----------|-----------|--------|------|-------|
| pcb_mobilenetv2_v1.keras | MobileNetV2 | 95.2% | 94.8% | 95.6% | 2026-02-07 | Initial model |
| pcb_mobilenetv2_v2.keras | MobileNetV2 | 97.1% | 96.9% | 97.3% | 2026-02-08 | Added dropout |

## Loading Saved Models

```python
from tensorflow.keras.models import load_model

# Load complete model
model = load_model('models/best_model.keras')

# Make predictions
predictions = model.predict(test_images)
```

## Model Checkpoints

Save checkpoints during training:

```python
from tensorflow.keras.callbacks import ModelCheckpoint

checkpoint = ModelCheckpoint(
    filepath='models/checkpoints/epoch_{epoch:02d}.keras',
    save_best_only=False,
    save_freq='epoch'
)

model.fit(train_gen, callbacks=[checkpoint])
```

## Best Practices

1. **Save best model only**:
```python
checkpoint = ModelCheckpoint(
    'models/best_model.keras',
    monitor='val_accuracy',
    save_best_only=True,
    mode='max'
)
```

2. **Version your models**: Include date or version number

3. **Document performance**: Keep track of metrics

4. **Save training history**:
```python
import json

with open('models/training_history_v1.json', 'w') as f:
    json.dump(history.history, f)
```

## Model Formats

### .keras (Recommended)
- New Keras format
- Includes architecture, weights, optimizer state
- Smaller file size
- Better compatibility

### .h5 (Legacy)
- HDF5 format
- Widely supported
- Larger file size

### SavedModel (For Deployment)
```python
# For TensorFlow Serving
model.export('models/saved_model/pcb_detector')
```

## Important Notes

⚠️ **DO NOT commit large model files to Git!**

Model files are excluded in `.gitignore`:
- `*.h5`
- `*.keras`
- `*.pb`
- `saved_models/`

### For Sharing Models:

1. **Small models (<100MB)**: Use Git LFS
2. **Large models**: Upload to:
   - Google Drive
   - Hugging Face Hub
   - AWS S3
   - Model registries

### Example: Sharing via Google Drive

```python
# After training, upload to Google Drive
# Share the link in README.md

# To download:
import gdown
gdown.download('GOOGLE_DRIVE_LINK', 'models/best_model.keras', quiet=False)
```

## Model Optimization

### Quantization (Reduce Size)

```python
import tensorflow as tf

# Convert to TFLite
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
tflite_model = converter.convert()

with open('models/model_quantized.tflite', 'wb') as f:
    f.write(tflite_model)
```

### Pruning (Reduce Parameters)

```python
import tensorflow_model_optimization as tfmot

# Apply pruning
prune_low_magnitude = tfmot.sparsity.keras.prune_low_magnitude
pruned_model = prune_low_magnitude(model)
```

## Model Evaluation

Always evaluate before deploying:

```python
# Evaluate on test set
results = model.evaluate(test_gen)
print(f"Test Accuracy: {results[1]:.4f}")

# Generate classification report
from sklearn.metrics import classification_report
y_pred = model.predict(test_gen)
y_true = test_gen.classes
print(classification_report(y_true, y_pred > 0.5))
```

## Export for Production

### ONNX Format (Cross-platform)

```bash
pip install tf2onnx
python -m tf2onnx.convert --keras models/best_model.keras --output models/model.onnx
```

### TensorFlow.js (Web Deployment)

```bash
pip install tensorflowjs
tensorflowjs_converter --input_format keras models/best_model.keras models/tfjs_model/
```

## Inference Example

```python
import numpy as np
from tensorflow.keras.preprocessing import image

# Load model
model = load_model('models/best_model.keras')

# Load and preprocess image
img = image.load_img('test_image.jpg', target_size=(224, 224))
img_array = image.img_to_array(img) / 255.0
img_array = np.expand_dims(img_array, axis=0)

# Predict
prediction = model.predict(img_array)[0][0]

if prediction > 0.5:
    print(f"DEFECT (confidence: {prediction:.2%})")
else:
    print(f"PASS (confidence: {(1-prediction):.2%})")
```

## Troubleshooting

### Error: "Unable to load model"
- Check TensorFlow version compatibility
- Ensure file path is correct
- Try loading with `compile=False`

### Error: "Out of Memory"
- Use smaller batch size
- Load only weights instead of full model
- Use model quantization

---

**Note**: This directory is excluded from version control. Only the structure is tracked via `.gitkeep` files.
