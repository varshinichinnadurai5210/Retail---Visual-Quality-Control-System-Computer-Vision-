# Contributing to PCB Defect Detection

First off, thank you for considering contributing to this project! It's people like you that make this project better.

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [How Can I Contribute?](#how-can-i-contribute)
- [Development Setup](#development-setup)
- [Pull Request Process](#pull-request-process)
- [Coding Standards](#coding-standards)

## Code of Conduct

This project and everyone participating in it is governed by a simple principle: **Be respectful and constructive**.

## How Can I Contribute?

### Reporting Bugs

Before creating bug reports, please check existing issues. When you create a bug report, include as many details as possible:

- **Use a clear and descriptive title**
- **Describe the exact steps to reproduce the problem**
- **Provide specific examples**
- **Describe the behavior you observed and what you expected**
- **Include screenshots if relevant**
- **Include your environment details** (OS, Python version, TensorFlow version)

### Suggesting Enhancements

Enhancement suggestions are tracked as GitHub issues. When creating an enhancement suggestion, include:

- **Use a clear and descriptive title**
- **Provide a detailed description of the suggested enhancement**
- **Explain why this enhancement would be useful**
- **List examples of how it would be used**

### Your First Code Contribution

Unsure where to begin? Look for issues labeled:
- `good first issue` - issues that should only require a few lines of code
- `help wanted` - issues that need extra attention

### Pull Requests

1. Fork the repository
2. Create a new branch (`git checkout -b feature/amazing-feature`)
3. Make your changes
4. Test your changes thoroughly
5. Commit your changes (`git commit -m 'Add amazing feature'`)
6. Push to the branch (`git push origin feature/amazing-feature`)
7. Open a Pull Request

## Development Setup

1. **Clone your fork**
```bash
git clone https://github.com/YOUR_USERNAME/pcb-defect-detection.git
cd pcb-defect-detection
```

2. **Create a virtual environment**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Create a new branch**
```bash
git checkout -b feature/your-feature-name
```

## Pull Request Process

1. **Update documentation** - Update README.md with details of changes if needed
2. **Update requirements** - Add any new dependencies to requirements.txt
3. **Test your changes** - Ensure code runs without errors
4. **Follow coding standards** - See below
5. **Write clear commit messages** - Use present tense ("Add feature" not "Added feature")
6. **One feature per PR** - Keep pull requests focused on a single feature or fix

### PR Template

When creating a PR, include:

```markdown
## Description
Brief description of changes

## Type of Change
- [ ] Bug fix
- [ ] New feature
- [ ] Documentation update
- [ ] Performance improvement

## Testing
How to test the changes

## Screenshots (if applicable)

## Checklist
- [ ] My code follows the project's coding standards
- [ ] I have tested my changes
- [ ] I have updated the documentation
- [ ] I have added comments to complex code
```

## Coding Standards

### Python Style Guide

Follow PEP 8 style guide:

```python
# Good
def calculate_accuracy(predictions, labels):
    """Calculate model accuracy."""
    correct = np.sum(predictions == labels)
    total = len(labels)
    return correct / total

# Use descriptive variable names
num_epochs = 20
learning_rate = 0.001

# Add docstrings to functions
def preprocess_image(image_path):
    """
    Load and preprocess an image for model input.
    
    Args:
        image_path: Path to the image file
        
    Returns:
        Preprocessed image array
    """
    # Implementation
    pass
```

### Code Organization

```python
# Imports at top, organized by:
# 1. Standard library
# 2. Third-party packages
# 3. Local modules

import os
import sys

import numpy as np
import tensorflow as tf

from utils import load_config
```

### Documentation

- Add docstrings to all functions and classes
- Use clear, descriptive variable names
- Comment complex logic
- Update README for new features

### Testing

- Test your code before submitting
- Include example usage in docstrings
- Verify notebook cells run in order

## Areas for Contribution

We're particularly interested in contributions in these areas:

1. **Model Improvements**
   - Experiment with different architectures
   - Hyperparameter optimization
   - Better augmentation strategies

2. **Visualization**
   - Better training visualizations
   - Grad-CAM or other explainability features
   - Interactive dashboards

3. **Documentation**
   - Tutorial notebooks
   - Video tutorials
   - API documentation

4. **Deployment**
   - Web interface
   - REST API
   - Mobile app integration
   - Docker containerization

5. **Dataset**
   - Data collection scripts
   - Additional augmentation techniques
   - Multi-class defect classification

6. **Performance**
   - Model optimization
   - Faster inference
   - Quantization for edge deployment

## Questions?

Feel free to open an issue with the label `question` if you need clarification on anything.

## Attribution

Contributors will be acknowledged in the README.md file.

---

Thank you for contributing! 🚀
