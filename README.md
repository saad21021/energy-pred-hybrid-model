# Energy Consumption Forecasting: Multi-Step Hybrid Model

This repository contains a state-of-the-art hybrid deep learning model for multi-step (24-hour) energy consumption forecasting using the UCI Household Power Consumption dataset.

## 🚀 Project Overview
The goal of this project is to predict the next 24 hours of global active power consumption based on historical data and current environmental/sensor factors. 

### Key Features
- **Multi-Step Forecasting**: Predicts a full 24-hour horizon in a single forward pass.
- **Hybrid Architecture**: Combines Convolutional Neural Networks (CNN) for feature extraction, Bidirectional LSTMs (BiLSTM) for temporal dependencies, and a Temporal Attention mechanism.
- **Dual Input Stream**: Integrates sequential history with simultaneous sensor readings (Voltage, Reactive Power, Sub-metering).
- **Advanced Preprocessing**: Implements log-transformation to handle right-skewed data and robust outlier removal.

## 🧠 Model Architecture
The "Best" model in this repository (`lstm_seminar_trying.ipynb`) utilizes:
1. **CNN Layers**: Extract local spatial-temporal patterns from the input sequence.
2. **Bidirectional LSTM**: Captures long-term dependencies from both past and future contexts within the window.
3. **Temporal Attention**: Dynamically weights the most important time steps in the past 24 hours to improve prediction accuracy for specific future hours.
4. **Dense Output**: A multi-head dense layer that outputs 24 discrete values corresponding to the next day's forecast.

## 📊 Performance
The model achieves high stability across the 24-hour horizon:
- **Hour 1 MAE**: ~0.19 kW
- **Hour 24 MAE**: ~0.43 kW
- **Overall MAE**: 0.4165 kW

## 🛠️ Requirements
To run the notebook, you will need:
- Python 3.8+
- TensorFlow 2.x
- Pandas, NumPy, Scikit-learn
- Matplotlib

## 📂 Dataset
The model is trained on the **UCI Individual Household Electric Power Consumption** dataset. 
> **Note**: The raw data file (`household_power_consumption.txt`) is ignored by git due to its size (>100MB). You can download it from the [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/individual+household+electric+power+consumption).

## 📝 Usage
1. Clone the repository.
2. Place `household_power_consumption.txt` in the root directory.
3. Open `lstm_seminar_trying.ipynb` and run all cells.
