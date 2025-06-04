# Algorithmic Trading System

## Overview
This project implements an algorithmic trading system for the BSE Ltd Stock using machine learning models. The system includes data processing, model training, backtesting, and visualization components.

## Features
- Data loading and preprocessing
- Feature engineering with technical indicators
- Model training using ensemble methods (Gradient Boosting, Random Forest, SVM, Logistic Regression, LSTM)
- Backtesting with risk management
- Visualization of trading results

## Setup
1. Clone the repository:
   ```bash
   git clone https://github.com/aagams2910/intraday-algorithmic-trading.git
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Ensure your data file (`BSE.csv`) is in the project directory.

## Usage
Run the main script:
```bash
python main.py
```

To retrain models:
```bash
python main.py --retrain
```

