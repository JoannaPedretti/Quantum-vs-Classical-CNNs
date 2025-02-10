# Quantum-vs-Classical-CNNs

## Description  
This repository contains the code files for my master's thesis research, which compares Quantum Convolutional Neural Networks (QCNN) to Classical Convolutional Neural Networks (CCNN). The objective of the project is to investigate if QCNNs are more robust to noisy, sparse, or limited input data compared to their classical counterparts. Artificial Gaussian noise is added to the dataset, and each model is trained with both clean and noisy data to analyze how added noise impacts performance metrics. The QCNN models are implemented using TensorFlow Quantum and Cirq, running on simulated quantum processors. All CCNN and QCNN models were trained on a T4 GPU from Google Colab.  

The research consists of three separate experiments:  
1. **Full-size multiclass classification (CCNN only)** – A classical CNN is trained on the full-resolution (28x28) Fashion-MNIST dataset using all ten classes to establish a baseline performance.  
2. **Multiclass classification on 4x4 images (CCNN vs. QCNN)** – Both CCNN and QCNN models are trained on a subset of the Fashion-MNIST dataset, where images are downscaled to 4x4 pixels to fit within the qubit constraints of the QCNN.  
3. **Binary classification on 4x4 images (CCNN vs. QCNN)** – Two out of ten classes are selected for binary classification, maintaining the 4x4 image size.  

Results from this research contribute to understanding the potential benefits of quantum machine learning for image classification tasks, particularly in situations where data may be noisy, corrupted, or limited in size.  

## Features
- Three .ipynb files originally written in Google Colab.
- Five CNN models total, three in CCNN file and two additional files for the QCNN models.
- Models are compared fairly by creating CCNN models with similar number of trainable parameters as the QCNN. 
- Train and evaluate each model first on clean data, then again on noisy data.
- Compute accuracy, loss, precision, recall, and F-1 score for each model and compare.
- Results are plotted in the respective notebook for each model. 

## Dataset
- [Fashion-MNIST dataset](https://www.kaggle.com/datasets/zalando-research/fashionmnist)
- 70,000 grayscale images of size 28x28, categorized into 10 classes, with 7,000 images per class.
- 60,000 training examples and 10,000 testing examples.

## Instructions to run the Notebooks
- Upload .ipynb files to Google Drive, open in Google Colab
- No setup required, just run the cells!
- Dataset is provided from tensorflow.keras.datasets
- Or run using Jupyter

## Summary of results after training for 10 epochs
### Full-Size Multiclass Classification (CCNN Only)  
| Model | Trainable Parameters | Accuracy (Clean Data) | Accuracy (Noisy Data) |
|--------|----------------------|----------------------|----------------------|
| CCNN | **721, 354** | **91.3%** | **83.8%** |

### Multiclass Classification on 4x4 Images (CCNN vs. QCNN)  
| Model | Trainable Parameters | Accuracy (Clean Data) | Accuracy (Noisy Data) |
|--------|----------------------|----------------------|----------------------|
| CCNN | **318** | **43.9%** | **30.6%** |
| QCNN | **290** | **40.5%** | **31.3%** |

**Observations:** The small image sizes and limited dataset had a clear negative impact on both models, but the QCNN performed slightly better in the presence of noisy data. 

### Binary Classification on 4x4 Images (CCNN vs. QCNN)  
| Model | Trainable Parameters | Accuracy (Clean Data) | Accuracy (Noisy Data) |
|--------|----------------------|----------------------|----------------------|
| CCNN | **97** | **69.9%** | **58.8%** |
| QCNN | **64** | **65.7%** | **62.7%** |

**Observations:** The QCNN model here had significantly more robustness to the noisy data compared to the CCNN.  
