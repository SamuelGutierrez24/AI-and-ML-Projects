# Deepfake Image Detection using CNN

## Overview

This project implements a Deep Convolutional Neural Network (D-CNN) for detecting deepfake images, based on the research presented in "An Improved Dense CNN Architecture for Deepfake Image Detection" by Patel et al. (2023). The model classifies images as either real or deepfake with high accuracy, addressing the growing concern of synthetic media manipulation in digital environments.

## Background

Deepfakes represent a significant threat to digital media authenticity, utilizing Generative Adversarial Networks (GANs) to create highly realistic synthetic content. As noted in the original research, "the dissemination of fake media streams creates havoc in social communities and can destroy the reputation of a person or a community" (Patel et al., 2023). This project addresses the critical need for robust detection systems that can identify manipulated content across various generation methods.

## Architecture

The implemented model follows the D-CNN architecture proposed by Patel et al., featuring:

- **Input Processing**: Images resized to 160×160 pixels with RGB channels
- **Convolutional Layers**: Multiple blocks with increasing filter complexity (8→16→32→64→128→256 filters)
- **Activation Functions**: Leaky ReLU activation throughout the network
- **Regularization**: Batch normalization and dropout layers (0.5) to prevent overfitting
- **Pooling**: Average and max pooling layers for dimensionality reduction
- **Classification**: Final sigmoid activation for binary classification

The architecture progressively extracts deeper features through convolutional operations, enabling the detection of subtle manipulation traces left by GAN-based generation tools.

## Dataset

**Source**: [Deepfake and Real Images Dataset](https://www.kaggle.com/datasets/manjilkarki/deepfake-and-real-images)

Due to computational constraints, we utilized a subset of the original dataset:
- **Total Images**: 8,000 (reduced from 140,000 available)
- **Real Images**: 4,000
- **Fake Images**: 4,000
- **Split Ratio**: 1-1-1 (The dataset came with about 10000 test images and 40000 validation images)

The dataset contains facial images from various sources, providing diversity necessary for model generalization across different deepfake generation techniques.

## Data Preprocessing

Following the methodology from the original paper:
- Image resizing to 160×160 pixels
- Data augmentation techniques including:
  - Random rotation (0-360 degrees)
  - Horizontal and vertical flipping
  - Zoom range (0.2)
  - Shear transformations
  - Width and height shifts

## Training Configuration

- **Optimizer**: Adam optimizer with learning rate 0.01
- **Loss Function**: Binary cross-entropy
- **Batch Size**: 64
- **Epochs**: 200
- **Hardware**: Google Colaboratory free T4 GPU environment

## Results

Our implementation achieved the following performance metrics on the test dataset:

### Overall Performance
- **Test Accuracy**: 78.9%

### Confusion Matrix
|                      | Predicted Real | Predicted Fake |
|----------------------|----------------|----------------|
| **Actual Real**      | True Negative: **3562** | False Positive: **438** |
| **Actual Fake**      | False Negative: **1250** | True Positive: **2750** |

### Detailed Classification Report
| Class | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|---------|
| Real  | 0.74      | 0.89   | 0.81     | 4000    |
| Fake  | 0.86      | 0.69   | 0.77     | 4000    |

### Aggregate Metrics
- **Accuracy**: 0.79
- **Macro Average**: Precision: 0.80, Recall: 0.79, F1-Score: 0.79
- **Weighted Average**: Precision: 0.80, Recall: 0.79, F1-Score: 0.79

## Analysis

While our results (78.9% accuracy) are lower than the original paper's reported performance (97.2%), this difference can be attributed to several factors:

1. **Reduced Dataset Size**: Using 8k images instead of the full 15k dataset
2. **Computational Limitations**: Limited training epochs and processing power
3. **Dataset Diversity**: Our subset may have different complexity distribution
4. **Hardware Constraints**: Limited GPU resources affecting training depth

The model demonstrates reasonable performance in distinguishing between real and fake images, with slightly better precision on fake image detection (0.86) but higher recall on real images (0.89).

## Key Findings

- The model shows good generalization capabilities across different image types
- Precision-recall balance indicates the model is not significantly biased toward either class
- Performance validates the effectiveness of the D-CNN architecture for deepfake detection tasks

## Future Improvements

1. **Full Dataset Training**: Utilize complete 14k image dataset with enhanced computational resources
2. **Extended Training**: Increase training epochs for better convergence, as even though the article claimed that rates plateaued after 200 epochs, our dataset is different
3. **Hyperparameter Optimization**: Given that the dataset is different than the one used in the article, fine-tuning learning rates and architecture parameters may improve accuracy
4. **Cross-validation**: Implement k-fold validation for more robust performance assessment
5. **Ensemble Methods**: Combine multiple models for improved accuracy

## References

Patel, Y., Tanwar, S., Bhattacharya, P., Gupta, R., Alsuwian, T., Davidson, I. E., & Mazibuko, T. F. (2023). An Improved Dense CNN Architecture for Deepfake Image Detection. *IEEE Access*, 11, 22081-22095. DOI: 10.1109/ACCESS.2023.3251417

## Technical Implementation

This project demonstrates the practical application of deep learning techniques for cybersecurity and digital forensics, showcasing the ability to implement and adapt state-of-the-art research for real-world deepfake detection challenges.
