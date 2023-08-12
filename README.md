# Stock-market-prediction-by-combining-CNNs-trained-on-multiple-time-frames

# Market Analysis in Forex Stock Market using Multi-CNN Model

This repository contains the implementation and research findings of a novel approach for market analysis in the Forex stock market. The study explores various econometric models, moving averages, technical indicators, and machine learning techniques commonly used for predicting stock market trends. The primary focus of this research is the introduction of a new model known as the **Multi-CNN (Convolutional Neural Network)** model, which incorporates domain knowledge specific to Forex trading.

## Abstract

This paper presents an innovative method for market analysis in the Forex stock market by introducing the **Multi-CNN model**, a deep learning approach that combines Convolutional Neural Networks (CNNs) and domain-specific knowledge of Forex trading. The model's performance is evaluated using EURUSD data spanning from January 2015 to December 2020. The dataset is meticulously preprocessed, normalized, and partitioned into distinct training, validation, and testing sets.

Benchmarked against established models including Single-LSTM, Single-GRU, and Single-CNN, the proposed Multi-CNN model demonstrates a notable enhancement in stock market trend prediction. This research provides compelling insights into the application of deep learning techniques for accurate forecasting of stock market trends. Notably, the advantages of integrating CNNs and leveraging multiple time frames are underscored, setting the Multi-CNN model apart from conventional models such as simple CNN, LSTM, and other recurrent neural network-based models.

## Key Contributions

- Introduction of the Multi-CNN model for Forex stock market analysis.
- Comprehensive evaluation using real-world EURUSD data from 2015 to 2020.
- Comparative analysis against Single-LSTM, Single-GRU, and Single-CNN models.
- Insights into the potential of deep learning techniques for precise stock market trend prediction.

## Repository Contents

- `data/`: Contains the preprocessed EURUSD dataset and train-validation-test splits.
- `notebooks/`: Jupyter notebook (`Multi_CNN_Network.ipynb`) detailing the implementation and experimentation process.
- `models/`: Saved Multi-CNN model checkpoints and other benchmark model checkpoints.
- `results/`: Quantitative and qualitative results of the Multi-CNN model's performance.
- `src/`: Source code for data preprocessing, model architecture, training, and evaluation.

## Usage

To reproduce the experiments and findings in this study, follow the steps outlined in the `Multi_CNN_Network.ipynb` notebook located in the `notebooks/` directory. The notebook provides a step-by-step walkthrough of data loading, preprocessing, model construction, training, and evaluation.

## Citation

If you find this work useful for your research or project, please consider citing:

