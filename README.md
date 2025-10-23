# CIFAR-10 CNN Image Classifier

A Convolutional Neural Network built from scratch to classify CIFAR-10 images into 10 categories. This project implements a custom CNN architecture with data augmentation.

## Project Highlights

### Architecture Details
1. **3 Convolutional Blocks:**
   - Block 1: 3 → 32 channels
   - Block 2: 32 → 64 channels
   - Block 3: 64 → 128 channels 
   - Each block: Conv2d → BatchNorm2d → ReLU → MaxPool2d

2. **Fully Connected Layers:**
   - Flatten: 4×4×128 → 2048 features
   - FC1: 2048 → 512 with ReLU and Dropout (50%)
   - FC2: 512 → 10 classes

3. **Total Parameters:** 1,147,914

### Model Highlights
1. **Data Augmentation** applied to training set:
   - Random horizontal flips
   - Random crops with padding (32×32 with 4px padding)
   - Normalization using CIFAR-10 statistics
2. **No augmentation** on validation/test sets for fair evaluation
3. **Batch Normalization** for training stability
4. **Dropout regularization** to prevent overfitting

## Dataset Details

- **CIFAR-10:** 60,000 32×32 color images in 10 classes
- **Training set:** 45,000 images (90% split)
- **Validation set:** 5,000 images (10% split)
- **Test set:** 10,000 images
- **Classes:** airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck
- **Normalization:** 
  - Mean: (0.4914, 0.4822, 0.4465)
  - Std: (0.2470, 0.2435, 0.2616)

## Repository Structure
```
cifar10-cnn/
├── data/
├── models/
│   └── best_model.pth
├── results/
│   ├── training_curves.png
│   ├── confusion_matrix.png
│   ├── per_class_accuracy.png
│   └── sample_predictions.png
├── src/
│   ├── config.py
│   ├── model.py
│   ├── dataset.py
│   ├── train.py
│   ├── evaluate.py
│   └── utils.py
├── test_dataset.py
├── test_model.py 
├── requirements.txt
└── README.md
```

## Training Setup

- **Optimizer:** Adam (lr=0.001, weight_decay=1e-4)
- **Loss Function:** CrossEntropyLoss
- **Batch Size:** 128
- **Epochs:** 50
- **Learning Rate:** Fixed at 0.001
- **Dropout Rate:** 0.5
- **Random Seed:** 42

## Performance

### Test Set Results (64.62% accuracy)

**Best Performing Classes:**
- Truck: 86% precision, 57% recall
- Ship: 82% precision, 74% recall
- Frog: 80% precision, 70% recall
- Horse: 73% precision, 65% recall

**Most Challenging Classes:**
- Cat: 42% precision, 43% recall
- Dog: 49% precision, 65% recall
- Bird: 55% precision, 51% recall

**Key Observations:**
- Model tends to over-predict "dog" class
- Vehicle classes (automobile, truck, ship) perform well
- Animal classes (cat, dog, bird) show more confusion
- 6.5× better than random guessing (10% baseline)

## Frameworks & Libraries

- **PyTorch** (≥2.0.0)
- **torchvision** (≥0.15.0)
- **matplotlib** (≥3.7.0)
- **scikit-learn** (≥1.3.0)
- **numpy** (≥1.24.0)
- **tqdm** (≥4.65.0)

## Workflow

1. **Data Loading:** Automatic CIFAR-10 download via torchvision
2. **Preprocessing:** Train/val split with different transforms
3. **Training:** 
   - Forward pass through CNN
   - CrossEntropyLoss calculation
   - Backpropagation with Adam optimizer
   - Validation after each epoch
   - Save best model based on validation accuracy
4. **Evaluation:**
   - Test on held-out 10,000 images
   - Generate confusion matrix
   - Calculate per-class metrics
   - Visualize sample predictions

## Visualizations

The evaluation script generates:
1. **Training Curves:** Loss and accuracy over 50 epochs
<img width="500" height="200" alt="training_curves" src="https://github.com/user-attachments/assets/5fd924c3-55d0-4b8b-97bf-e847ce016dd4" />


2. **Confusion Matrix:** 10×10 heatmap showing class confusions
<img width="500" height="200" alt="confusion_matrix" src="https://github.com/user-attachments/assets/b8da9c78-5f6b-4a75-a7b0-337d2a993a69" />


3. **Per-Class Accuracy:** Bar chart for each of 10 classes
<img width="500" height="200" alt="per_class_accuracy" src="https://github.com/user-attachments/assets/4c261ed7-16cc-4588-80c9-43a7844d3ded" />


4. **Sample Predictions:** Grid of 20 images with true/predicted labels (green=correct, red=incorrect)
<img width="500" height="200" alt="sample_predictions" src="https://github.com/user-attachments/assets/d8ac7a97-cd57-4c6d-82d6-834e1c6a9bcc" />


## Key Learnings

1. **Data augmentation is critical** for small datasets like CIFAR-10
2. **Batch normalization** significantly stabilizes training
3. **Dropout regularization** helps prevent overfitting
4. **Professional project structure** makes debugging and iteration easier
5. **Checkpoint saving** enables recovery from interruptions
6. **Validation monitoring** prevents overfitting to training data

## Improvements & Future Work

1. **Learning Rate Scheduling:** Implement ReduceLROnPlateau or CosineAnnealing
2. **Architecture Improvements:**
   - Add residual connections (ResNet-style)
   - Experiment with deeper networks
   - Try different filter sizes
3. **Optimizer Experiments:** Compare Adam vs SGD with momentum
4. **Transfer Learning:** Fine-tune pre-trained models (ResNet, VGG)
5. **Ensemble Methods:** Combine multiple models for better accuracy
6. **MPS Support:** Fix BatchNorm compatibility issues for M4 GPU training
7. **Hyperparameter Tuning:** Systematic search for optimal learning rate, dropout, etc.

## Assumptions

1. **CIFAR-10 normalization statistics** are pre-calculated and fixed
2. **90/10 train/validation split** is reasonable for 50k training images
3. **Batch size of 128** fits in available memory
4. **50 epochs** is sufficient for convergence
5. **Adam optimizer** works well without tuning 

## Known Issues

1. **Validation accuracy bug:** Shows unrealistic 99%+ during training
2. **Suboptimal test accuracy:** 64.62% is below typical CIFAR-10 CNNs (~75-80% expected)

## Key Observations

- **Data augmentation complexity:** Random crops and flips make training harder but improve generalization
- **Class imbalance in difficulty:** Animals (cat, dog, bird) are harder than vehicles (ship, truck)
- **Training on CPU:** Takes 3-4× longer than GPU but produces identical results
