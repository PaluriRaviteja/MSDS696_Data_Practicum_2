# Recognizing Hand-Drawn Fruit Sketches Using Machine Learning

A Convolutional Neural Network (CNN) implementation that classifies simple hand-drawn fruit sketches with 94% accuracy, demonstrating that machines can learn to interpret abstract, childlike drawings with minimal training data.

**Course:** MSDS692 – Data Science Practicum  
**Institution:** Regis University  
**Author:** Raviteja Paluri

---

## Project Overview

This project explores whether machine learning models can recognize hand-drawn fruit sketches as intuitively as humans do. Unlike traditional computer vision systems that rely on color, texture, and photographic detail, this model interprets simple, abstract sketches based purely on shapes and outlines.

**Key Achievement:** Starting with just 50 original hand-drawn sketches, the model achieved 94% validation accuracy through intelligent data augmentation and a carefully designed CNN architecture.

**Classes:** Apple, Banana, Grapes, Pineapple, Strawberry

---

## Why This Project Matters

Hand-drawn sketches present unique challenges for computer vision:

1. **High Variability:** No two hand-drawn sketches are identical—stroke thickness, proportions, and style vary significantly across individuals.

2. **Shape Abstraction:** Childlike drawings simplify reality into symbolic representations, lacking the rich features (color, texture, shading) that traditional CV systems depend on.

3. **Limited Data Availability:** Unlike photo datasets with millions of images, hand-drawn sketch datasets are scarce and expensive to collect.

This project demonstrates that with proper preprocessing, data augmentation, and architecture design, deep learning models can achieve strong performance even with minimal training data—an important consideration for real-world applications where data collection is costly.

**Potential Applications:**
- Educational tools for children learning to draw
- UI/UX sketch-to-design conversion systems
- Accessibility applications for alternative input methods
- Quick-sketch search engines

---

## Dataset

### Source
- **Original Data:** 50 hand-drawn fruit sketches (10 per class)
- **Drawing Style:** Simple, QuickDraw-inspired sketches
- **Format:** PNG images, grayscale
- **Classes:** 5 balanced classes (apple, banana, grapes, pineapple, strawberry)

### Data Augmentation
To overcome the challenge of limited training data, the original 50 sketches were expanded to approximately 5,000 images through on-the-fly augmentation during training:

- **Rotation:** ±20° random rotation
- **Translation:** ±15% horizontal and vertical shifts
- **Zoom:** ±20% random scaling
- **Shear:** ±15% shear transformation
- **Horizontal Flip:** 50% probability

This augmentation strategy teaches the model to be invariant to position, orientation, and scale—critical for handling real-world sketch variations.

### Train/Validation Split
- **Training:** 80% (with augmentation)
- **Validation:** 20% (with augmentation)

---

## Preprocessing & Feature Engineering

Every image undergoes a standardized preprocessing pipeline to ensure consistency between training and inference:

### 1. Grayscale Conversion
Color information is removed, focusing the model on shapes and contours rather than color cues that don't exist in simple sketches.

### 2. Resize to 64×64
All images are standardized to 64×64 pixels, providing a consistent input dimension while keeping computational requirements low.

### 3. Binary Thresholding
A threshold value of 150 is applied to convert images to binary (black and white), emphasizing edges and reducing noise from scanning artifacts or drawing pressure variations.

### 4. Normalization
Pixel values are normalized to the range [0, 1] by dividing by 255, improving training stability and convergence speed.

### Pipeline Summary
```
Raw Sketch → Grayscale → Resize (64×64) → Threshold (150) → Normalize → Model Input
```

This consistent preprocessing is critical—even small deviations between training and inference preprocessing can significantly degrade model performance.

---

## Baseline Models

Before developing the CNN, baseline performance was established:

### Random Guess Baseline
- **Accuracy:** 20% (1 in 5 classes)
- **Method:** Random selection among 5 classes

### Majority Class Baseline
- **Accuracy:** ~20% (balanced dataset)
- **Method:** Always predict the most common class

**Baseline Comparison:** The CNN's 94% accuracy represents a 4.7× improvement over random guessing, demonstrating that the model has learned meaningful patterns rather than memorizing noise.

---

## Model Development

### Architecture

A Convolutional Neural Network was designed with three convolutional blocks followed by dense classification layers:

```
Input Layer (64×64×1 grayscale images)
    ↓
Conv2D Block 1: 16 filters (3×3) + ReLU
    → MaxPooling2D (2×2)
    ↓
Conv2D Block 2: 32 filters (3×3) + ReLU
    → MaxPooling2D (2×2)
    ↓
Conv2D Block 3: 64 filters (3×3) + ReLU
    → MaxPooling2D (2×2)
    ↓
Flatten Layer
    ↓
Dense Layer: 64 units + ReLU
    → Dropout (0.3)
    ↓
Output Layer: 5 units + Softmax
```

### Design Rationale

**Progressive Filter Increase (16→32→64):**  
Early layers detect simple features (edges, curves), while deeper layers combine these into complex patterns (pineapple crosshatch, banana crescent shape).

**Max Pooling:**  
Reduces spatial dimensions while retaining the most important features, improving computational efficiency and translation invariance.

**Dropout (0.3):**  
Prevents overfitting by randomly deactivating 30% of neurons during training, forcing the network to learn robust, generalizable features.

**Lightweight Design:**  
With only three convolutional blocks, the model trains quickly (5-10 minutes on CPU) while still achieving high accuracy—ideal for resource-constrained environments.

### Training Strategy

**Optimizer:** Adam (adaptive learning rate)  
**Loss Function:** Categorical Crossentropy (multi-class classification)  
**Batch Size:** 8 (small batch size works well with limited data)  
**Epochs:** 10 (increased from initial experiments)  
**Callbacks:** Early stopping with patience to prevent overtraining

### Hyperparameter Choices

The architecture was deliberately kept simple to:
1. Minimize training time with limited data
2. Reduce risk of overfitting
3. Enable fast inference for real-time GUI application
4. Demonstrate that effective models don't require massive complexity

---

## Results

### Model Performance

**Final Validation Accuracy:** 94%

<img width="1091" height="500" alt="Screenshot 2025-12-15 at 12 04 19 PM" src="https://github.com/user-attachments/assets/f38f9d35-c216-4119-a020-943241f88358" />


This represents strong generalization from just 50 original training examples, demonstrating the effectiveness of data augmentation and architectural choices.

### Confidence Analysis

**High Confidence Predictions (>90%):**  
Sketches with distinctive, unique features like pineapple crosshatch patterns or well-defined shapes achieve near-perfect confidence.

**Example:** Pineapple sketch with crosshatch body and spiky crown → 97.6% confidence

**Medium Confidence Predictions (50-70%):**  
Reasonable sketches with some ambiguity, such as simple fruit outlines without distinctive details.

**Example:** Grape cluster (varies by drawing style) → 53.6% confidence

**Low Confidence Predictions (<50%):**  
Very simple or ambiguous shapes that could represent multiple fruits.

**Example:** Simple crescent curve → 46% confidence (banana, but uncertain)

### Prediction Examples

| Sketch Description | Prediction | Confidence | Analysis |
|-------------------|------------|------------|----------|
| Crosshatch oval with spiky crown | Pineapple | 97.6% | Distinctive features enable high confidence |
| Cluster of circles with stem | Grapes | 53.6% | Pattern correctly identified |
| Simple crescent shape | Banana | 46.0% | Appropriate uncertainty for minimal detail |
| Round shape with internal circles | Grapes | 100.0% | Model weighted internal pattern heavily |

<img width="587" height="584" alt="Screenshot 2025-12-11 at 8 51 11 PM" src="https://github.com/user-attachments/assets/8fce0b8f-afd9-42f2-b8f2-9abe29de0069" />

<img width="587" height="584" alt="Screenshot 2025-12-11 at 8 50 19 PM" src="https://github.com/user-attachments/assets/f91dc917-59ec-4c7a-9357-9c2e29b41548" />

<img width="587" height="584" alt="Screenshot 2025-12-11 at 8 49 35 PM" src="https://github.com/user-attachments/assets/683bf481-8d1c-4604-8f48-eeecde4a8fcc" />


### Model Insights

**What the Model Learned:**
- Pineapples have unique crosshatch patterns and spiky crowns
- Grapes appear as clusters of circular elements
- Bananas are crescent-shaped curves
- Distinctive features correlate strongly with confidence scores

**Confidence Calibration:**  
The model demonstrates appropriate uncertainty on ambiguous sketches rather than overconfident incorrect predictions—a desirable property in production systems.

---

## GUI Demo

An interactive Tkinter-based GUI application enables real-time sketch recognition.
<img width="580" height="575" alt="Screenshot 2025-12-15 at 12 10 45 PM" src="https://github.com/user-attachments/assets/b425fc64-8e93-46d1-bd65-d9ce9c51a3b0" />


### Features

**1. Drawing Canvas**  
Users can draw fruit sketches directly using mouse or trackpad on a 400×400 pixel canvas.

**2. Image Upload**  
Alternative input method for users who prefer to upload pre-drawn sketches (PNG/JPG supported).

**3. Real-Time Prediction**  
Instant classification with confidence scores displayed prominently.

**4. Sample Viewer**  
Shows random training examples to guide users on appropriate drawing styles.

**5. Debug Output**  
Automatically saves preprocessed images for troubleshooting and transparency.

### User Experience

The interface provides helpful guidance: *"Draw simple & fast (QuickDraw style)"* to set appropriate expectations and improve prediction accuracy.

### Technical Implementation

The GUI replicates the exact preprocessing pipeline used during training:
- Converts drawings to grayscale
- Resizes to 64×64 pixels
- Applies binary thresholding (150)
- Normalizes to [0, 1]
- Passes to trained model for inference

**Inference Time:** <100ms per prediction, enabling smooth interactive experience.

---

## Project Limitations

### 1. Dataset Scale
With only 50 original sketches (10 per class), the model's exposure to diverse drawing styles is limited. While data augmentation helps, it cannot fully replicate the natural variation in human sketches.

**Impact:** Model may struggle with drawing styles significantly different from training examples.

### 2. Sketch Variability
Hand-drawn sketches vary enormously across individuals—some use bold strokes, others sketch lightly; some draw geometrically, others organically. The current dataset may not capture this full spectrum.

**Impact:** Performance may degrade on sketches from populations underrepresented in training data.

### 3. Visual Ambiguity
Some sketches genuinely combine features of multiple fruits (e.g., round shape with stem could be apple or grapes). The model must make deterministic predictions even when multiple interpretations are valid.

**Impact:** High-confidence misclassifications on ambiguous inputs that even humans might debate.

### 4. Limited Class Set
Only 5 fruit classes are supported. Real-world applications would require dozens or hundreds of object categories.

**Impact:** Practical utility is limited to educational demonstrations rather than production use.

### 5. Lack of Explainability
The model provides predictions and confidence scores but doesn't show which parts of the sketch influenced its decision.

**Impact:** Users cannot understand why predictions are made, limiting trust and debugging capabilities.

<img width="438" height="543" alt="Screenshot 2025-12-11 at 1 44 53 AM" src="https://github.com/user-attachments/assets/d4ab13e5-a8cc-41ed-87fd-3841cd27c3a2" />

---

## Future Work

### Short-Term Improvements

**1. Expand Dataset**  
Collect 100-200 additional sketches per class from diverse participants to capture more drawing styles. This would directly address the dataset scale limitation.

**2. Add More Fruit Classes**  
Expand to 10-15 classes (orange, watermelon, cherry, etc.) to test scalability and identify confusable category pairs.

**3. Implement Grad-CAM Visualization**  
Use Gradient-weighted Class Activation Mapping to generate heatmaps showing which sketch regions influenced predictions. This addresses the explainability limitation.

### Medium-Term Improvements

**4. Multi-Label Classification**  
Enable the model to output multiple predictions with probabilities when sketches are genuinely ambiguous (e.g., 45% apple, 42% grapes, 13% other).

**5. Web Deployment**  
Convert GUI to Flask/Streamlit web application, making it accessible without local installation.

**6. Mobile App Development**  
Deploy model using TensorFlow Lite for iOS/Android applications, enabling on-device inference.

### Long-Term Vision

**7. Few-Shot Learning**  
Implement meta-learning approaches to add new fruit categories with just 5-10 examples, reducing data collection burden.

**8. Generalize Beyond Fruits**  
Extend to other object categories (animals, vehicles, household items) to create a general-purpose sketch recognition system.

**9. Collaborative Dataset Creation**  
Build a web platform where users worldwide can contribute sketches, creating a large-scale, diverse dataset while using the model.

---

## How to Run the Project

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/yourusername/fruit-sketch-recognizer.git
cd fruit-sketch-recognizer
```

2. **Install dependencies**
```bash
pip install tensorflow keras numpy pillow opencv-python matplotlib scikit-learn
```

### Running the GUI Application

```bash
python fruit_gui.py
```

This launches the interactive application where you can:
- Draw sketches using your mouse
- Upload pre-drawn images
- See real-time predictions with confidence scores

### Training the Model

To retrain the model from scratch:

```bash
python train_fruit_cnn.py
```

**Requirements:**
- `data_preprocessed/` folder with training images organized by class
- Each class subfolder should contain preprocessed 64×64 grayscale PNG images

**Training Output:**
- Model saved as `fruit_cnn.h5`
- Training metrics printed to console
- ~5-10 minutes on modern CPU

### Project Structure

```
fruit-sketch-recognizer/
├── train_fruit_cnn.py          # Model training script
├── fruit_gui.py                # Interactive GUI application
├── fruit_cnn.h5                # Trained model weights
├── data_preprocessed/          # Training data (64×64 grayscale)
│   ├── apple/
│   ├── banana/
│   ├── grapes/
│   ├── pineapple/
│   └── strawberry/
└── README.md                  
```

---

## Technologies Used

**Deep Learning Framework:**
- TensorFlow 2.x
- Keras Sequential API

**Image Processing:**
- OpenCV (cv2)
- Pillow (PIL)

**GUI Development:**
- Tkinter

**Data Manipulation:**
- NumPy
- Scikit-learn

**Development Environment:**
- Python 3.8+


---
