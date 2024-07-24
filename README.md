# 🎵 Audio Genre Classification 🎶

This project involves classifying audio files into various genres using machine learning and deep learning techniques. The dataset used consists of audio features extracted from music files.

---

## 📋 Prerequisites

Ensure you have the following packages installed:
- `xgboost`
- `librosa`
- `hyperopt`
- `pandas`
- `numpy`
- `scipy`
- `scikit-learn`
- `matplotlib`
- `seaborn`
- `keras`
- `tensorflow`

---

## 📂 Dataset

The dataset consists of audio files in the `.wav` format, with features extracted for each file. The features include chroma, spectral, and MFCC (Mel-frequency cepstral coefficients) among others.

---

## 📊 Findings

### 🔍 Data Insights

1. **Feature Distribution**:
    - The features extracted from the audio files, such as chroma, spectral, and MFCC, show significant variation across different genres. This variation is crucial for distinguishing between genres.

2. **Label Distribution**:
    - The dataset contains a balanced distribution of labels across genres, which is beneficial for training robust classification models.

---

### 🚀 Model Performance

1. **XGBoost Classifier**:
    - **Training Accuracy**: 99.92%
    - **Testing Accuracy**: 91.59%
    - **Observations**: The XGBoost classifier performs exceptionally well on the training data, achieving near-perfect accuracy. The testing accuracy is also high, indicating good generalization capability. Some genres, however, exhibit slightly lower precision and recall, suggesting room for improvement in distinguishing those genres.

2. **Convolutional Neural Network (CNN)**:
    - **Training Accuracy**: The model achieves high accuracy over multiple epochs, with validation accuracy stabilizing around 92.89%.
    - **Loss Trends**: The training and validation loss decrease consistently over epochs, indicating effective learning and minimal overfitting.
    - **Observations**: The CNN model, with multiple dense layers and dropout for regularization, performs robustly on the genre classification task. The use of dropout layers helps in preventing overfitting, leading to better generalization on the validation data.

---

### 📈 Evaluation Metrics

1. **Confusion Matrix**:
    - The confusion matrix reveals that the models correctly classify most of the genres with high accuracy. Misclassifications are minimal but tend to occur more between genres with similar audio characteristics.

2. **Classification Report**:
    - Precision, recall, and F1-scores are generally high across all genres. Certain genres, such as `rock` and `hiphop`, show slightly lower scores, suggesting that these genres may have overlapping features that make them harder to distinguish.

---

### 📊 Visualizations

1. **Spectrograms and Waveforms**:
    - Visual inspection of spectrograms and waveforms for different genres highlights distinct patterns that align with the quantitative features extracted. This visual differentiation supports the model's ability to classify genres based on learned features.

2. **Correlation Heatmap**:
    - The correlation heatmap of features reveals strong correlations among certain features, which the models leverage for classification. Understanding these correlations helps in feature selection and model refinement.

---

## 📌 Conclusions

- The project successfully demonstrates the use of both machine learning (XGBoost) and deep learning (CNN) techniques for audio genre classification.
- The high accuracy and robust performance of the models indicate that the extracted features effectively capture the characteristics of different music genres.
- Future work could involve exploring more advanced architectures, such as recurrent neural networks (RNNs) or attention mechanisms, to further improve classification accuracy and handle more complex audio patterns.

---

## 💬 Acknowledgements

I would like to acknowledge the authors of the libraries and tools used in this project. Their work has made this project possible.
