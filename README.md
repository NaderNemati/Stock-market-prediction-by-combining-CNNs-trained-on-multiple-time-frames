# Market Analysis in Forex Stock Market using Multi-CNN Model

This repository contains the implementation and research findings of a novel approach for market analysis in the Forex stock market. The study explores various econometric models, moving averages, technical indicators, and machine learning techniques commonly used for predicting stock market trends. The primary focus of this research is the introduction of a new model known as the **Multi-CNN (Convolutional Neural Network)** model, which incorporates domain knowledge specific to Forex trading.

## Abstract

This paper presents an innovative method for market analysis in the Forex stock market by introducing the **Multi-CNN model**, a deep learning approach that combines Convolutional Neural Networks (CNNs) and domain-specific knowledge of Forex trading. The model's performance is evaluated using EURUSD data spanning from January 2015 to December 2020. The dataset is meticulously preprocessed, normalized, and partitioned into distinct training, validation, and testing sets.

Benchmarked against established models including Single-LSTM, Single-GRU, and Single-CNN, the proposed Multi-CNN model demonstrates a notable enhancement in stock market trend prediction. This research provides compelling insights into the application of deep learning techniques for accurate forecasting of stock market trends. Notably, the advantages of integrating CNNs and leveraging multiple time frames are underscored, setting the Multi-CNN model apart from conventional models such as simple CNN, LSTM, and other recurrent neural network-based models.

## Model Description

### Suggested Model Architecture

To enhance the accuracy of EURUSD predictions, this academic paper introduces innovative models that exploit multiple time frames of EURUSD data during the training process. The three distinct time frames employed in the proposed model are carefully selected: 15 minutes, 30 minutes, and 1 hour. Incorporating these diverse time frames allows the model to capture intricate and intricate hidden patterns inherent in sequential data, particularly in the context of time series analysis.

#### Network Composition

Each model in this study is composed of three separate neural networks, each utilizing a pair of convolutional neural network (CNN) layers. The first CNN layer is designed with 256 neurons, enabling it to extract high-level features from the input data. The subsequent layer, comprising 128 neurons, further refines the extracted features, effectively capturing the underlying patterns. This hierarchical architecture facilitates the learning of intricate representations that are crucial for accurate predictions.

Following the convolutional layers, a fully connected layer with four neurons is employed in each neural network. This layer is responsible for predicting the values associated with open, high, low, and close prices for each specific time frame. By producing predictions at multiple levels, the model can effectively capture and leverage the inherent characteristics of each time frame, leading to more robust predictions.

#### Fusion of Information

To consolidate the predicted values obtained from the different time frames and combine the CNNs, a concatenation layer is employed, enabling the fusion of information across the temporal spectrum. Subsequently, a fully connected layer is utilized to predict the output value for the target timeframe. This design architecture ensures the comprehensive integration of information from diverse time frames, facilitating a holistic understanding of the data and enhancing the model’s predictive performance.

For an in-depth understanding of the proposed models and their detailed specifications, refer to Table I in this paper. Furthermore, Figure 3 visually presents the architecture of the Multi-CNN model, which has demonstrated superior performance in terms of loss reduction when compared to alternative models.

### Illustrative Figures and Tables

For visual representation of the model architecture and comprehensive details about the suggested models, please refer to the following resources:

- **Table I:** Detailed specifications of the proposed models.
- **Figure 3:** Visual depiction of the Multi-CNN model architecture, showcasing its superior performance in loss reduction compared to alternative models.

These resources provide a clear understanding of the model's composition and its effectiveness in capturing intricate patterns for improved prediction accuracy.

## Conclusion

In conclusion, this study presents the Multi-CNN model, a groundbreaking artificial neural network architecture that outperforms traditional 1D CNN-based approaches in the realm of Forex price prediction. Through comprehensive evaluation on the EURUSD dataset, the Multi-CNN model demonstrates remarkable performance enhancements when compared to LSTM, GRU, and 1D CNN architectures.

The superiority of the Multi-CNN model is evidenced by its significant reduction in both Mean Square Error and Forex Loss Function, underscoring its capacity for making more accurate predictions. Notably, the Multi-CNN models employing FLF and MSE as cost functions consistently exhibit lower loss values in contrast to alternative methodologies. This consistent performance underscores the model's efficacy in minimizing prediction errors and capturing the intricate underlying patterns inherent in Forex price data.

The advancements achieved in this study hold substantial implications for the field of Forex market trend forecasting, elevating its accuracy and dependability. By embracing the Multi-CNN model, traders and investors stand to gain from enhanced predictive capabilities in the dynamic landscape of Forex trading. This paves the way for further exploration and innovation in Forex price prediction techniques, ultimately empowering decision-makers with greater insights and the potential for heightened profitability.

The findings and methodologies presented in this research contribute to a deeper understanding of predictive modeling in financial markets and emphasize the potential of advanced deep learning techniques in refining trading strategies and informed decision-making.

We invite researchers, practitioners, and enthusiasts to delve into the details of this study and explore the opportunities it presents for refining and expanding the domain of Forex price prediction, fostering ongoing advancements and collaboration in this exciting field.


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


## License

This project is licensed under the [MIT License](LICENSE).

---

For any inquiries or collaborations, please contact [Your Name](mailto:your.email@example.com).


