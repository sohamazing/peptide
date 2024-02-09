# Peptide Hydrophobicity Prediction using Neural Networks

This repository contains a Python script for building and training a neural network model to predict hydrophobicity values based on amino acid sequences. The model is implemented using TensorFlow and Keras.

## Overview

Hydrophobicity is an essential property of biomolecules, and predicting it can provide insights into the structure and function of proteins. This program utilizes neural networks to predict hydrophobicity values from amino acid sequences.

## Requirements

- Python 3.x
- TensorFlow
- pandas
- numpy
- matplotlib

Install the required packages using:

```bash
pip install -r requirements.txt
```

## Dataset
The dataset used for training the model is stored in the file HumanPlasma2023-04.csv. It contains amino acid sequences and corresponding hydrophobicity values.

## Usage
# Clone the repository:

```bash
git clone https://github.com/your-username/hydrophobicity-prediction.git
cd hydrophobicity-prediction
```

# Install dependencies:

```bash
pip install -r requirements.txt
```

# Run the script:
```bash
python3 hydrophobicity.py
```

## Model Architecture
The neural network model consists of multiple dense layers with dropout and regularization for hydrophobicity prediction. The architecture details can be found in the script hydrophobicity.py.

## Results
The model is trained and evaluated on a dataset split into training, validation, and test sets. The evaluation metrics include Mean Squared Error (MSE) and Mean Absolute Error (MAE), providing insights into the model's performance.

## Visualizations
The script generates a plot illustrating the training and validation loss curves over epochs. This visualization helps in assessing the training progress and potential overfitting.

