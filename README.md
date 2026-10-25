# EEG Motor Imagery Classification Using CSP + LDA and CNNs

## Overview

This project investigates the classification of left-hand versus right-hand motor imagery from electroencephalography (EEG) signals.

Using the PhysioNet EEG Motor Movement/Imagery dataset, I implemented and compared a traditional machine learning pipeline based on Common Spatial Patterns (CSP) and Linear Discriminant Analysis (LDA) with deep learning approaches based on Convolutional Neural Networks (CNNs).

The objective was to evaluate the classification performance of these approaches under participant-wise cross-validation, where participants in the test fold are excluded from model training.

## Research Objectives

- Preprocess EEG signals for motor imagery classification.
- Extract discriminative spatial features using Common Spatial Patterns.
- Classify CSP features using Linear Discriminant Analysis.
- Implement a CNN for learning EEG representations directly from signals.
- Investigate regularization and early stopping to reduce overfitting.
- Compare model performance using accuracy and balanced accuracy.

## Dataset

**Dataset:** PhysioNet EEG Motor Movement/Imagery Dataset (EEGMMIDB)

The experiments use recordings from the imagined left- and right-fist movement task.

Dataset source: https://physionet.org/content/eegmmidb/

The dataset was processed to construct trials containing 64 EEG channels and 481 time samples per trial.

Final processed dataset:
- Total trials: 1,637
- EEG channels per trial: 64
- Time samples per trial: 481
- Left-hand imagery trials: 824
- Right-hand imagery trials: 813
- Unique participants represented: 109

The original dataset is not included in this repository. Users should obtain it from the original source and follow its applicable data-use terms.

## Methodology

### 1. EEG Preprocessing

The preprocessing pipeline includes:

- Resampling recordings to 160 Hz.
- Band-pass filtering between 8 and 30 Hz.
- Epoching the EEG signals from 0.5 to 3.5 seconds relative to the selected event markers.
- Constructing a binary classification dataset for left- versus right-hand motor imagery.

Each trial is represented as a matrix of shape `(64, 481)`.

### 2. CSP + LDA

Common Spatial Patterns (CSP) is used to learn spatial filters that emphasize differences in signal variance between the two classes.

Log-variance features are then extracted from the CSP components and classified using Linear Discriminant Analysis.

The initial baseline uses four CSP components.

### 3. CNN

A one-dimensional CNN is trained directly on the multichannel EEG trials.

The architecture consists of:
- Temporal convolutional layers.
- Batch normalization and ReLU activations.
- Max-pooling layers.
- Adaptive average pooling.
- Dropout and a final linear classification layer.

The model is trained using Adam and cross-entropy loss.

### 4. CNN With Regularization and Early Stopping

A second CNN experiment introduces stronger weight decay and early stopping based on validation loss.

A participant-wise validation split is used within the training data, while the outer test participants remain held out for evaluation.

### 5. Evaluation Strategy

The models are evaluated using five-fold GroupKFold cross-validation, grouping trials by participant.

This design evaluates generalization to participants not included in the corresponding training fold.

Metrics:
- Accuracy
- Balanced accuracy

## Results

Mean performance across five participant-wise cross-validation folds:

| Model | Mean Accuracy | Mean Balanced Accuracy |
|---|---:|---:|
| CSP + LDA | 58.84% | 58.83% |
| CNN | 54.73% | 54.83% |
| CNN with early stopping and stronger weight decay | 56.22% | 56.38% |

### Key Observations

- CSP + LDA achieved the highest mean accuracy among the evaluated models.
- The original CNN showed a substantial gap between training and test accuracy, suggesting overfitting.
- Early stopping and stronger weight decay reduced the training burden and improved mean test accuracy relative to the original CNN.
- The CNN with early stopping did not outperform the CSP + LDA baseline.

These results are preliminary and specific to the selected preprocessing, architectures, hyperparameters, and cross-validation procedure. They do not establish that one approach is universally superior.

## Technologies Used

- Python
- NumPy
- pandas
- MNE-Python
- scikit-learn
- PyTorch
- Matplotlib
- Jupyter Notebook

## Reproducibility

1. Clone this repository.
2. Install the required Python dependencies.
3. Obtain the dataset from PhysioNet.
4. Configure the dataset path in the preprocessing notebook or script.
5. Run the notebooks in the documented order.
6. Evaluate the models using the participant-wise cross-validation procedure.

See `requirements.txt` for the required packages.

## Limitations

- The experiments use a selected motor imagery task and preprocessing configuration.
- CNN performance may depend strongly on architecture, hyperparameters, and regularization.
- Results are preliminary and should be interpreted in the context of the evaluation protocol.
- Further experiments are required to assess statistical significance and determine whether alternative CSP configurations or EEG-specific architectures improve performance.

## Future Work

Potential extensions include:
- Comparing different numbers of CSP components.
- Evaluating EEG-specific architectures such as EEGNet.
- Investigating additional preprocessing and feature extraction methods.
- Performing statistical comparisons across paired cross-validation folds.
- Exploring subject-specific versus cross-subject classification.

## Acknowledgements

This project uses data from the PhysioNet EEG Motor Movement/Imagery Dataset.

Dataset: https://physionet.org/content/eegmmidb/
# eeg-motor-imagery-classification
