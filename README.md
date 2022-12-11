# ABSA-PyTorch

This project recreates the ABSA-PyTorch repository for aspect-based sentiment analysis using PyTorch.

## Project Overview

The goal is to replicate and potentially improve upon the original repository's functionality, with a strong emphasis on modularity, readability, and maintainability.

## Directory Structure

- `src`: Source code directory.
  - `data`: Data loading and preprocessing module.
  - `models`: Model definition module.
  - `training`: Training and evaluation module.
  - `utils`: Utility functions and helper classes.
- `data`: Directory for storing datasets.
- `checkpoints`: Directory for saving model checkpoints.
- `logs`: Directory to store training and evaluation logs.

## Key Files

- `src/data/dataset.py`: Defines the dataset classes.
- `src/data/dataloaders.py`: Defines the data loaders.
- `src/data/vocabulary.py`: Handles vocabulary creation.
- `src/models/model.py`: Defines the neural network model.
- `src/training/trainer.py`: Implements the training loop.
- `src/utils/config.py`: Handles configuration loading.
- `train.py`: Main training script.
- `evaluate.py`: Main evaluation script.
