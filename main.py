import pandas as pd
import numpy as np
from datetime import datetime
import argparse
import data_processor
import models
import backtester
import visualizer
import config
import os

def main():
    """Main function to run the trading system"""
    # Parse command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--retrain', action='store_true', help='Retrain models')
    args = parser.parse_args()
    
    print("Loading and preparing data...")
    df = data_processor.load_data(config.DATA_FILE)
    df = data_processor.prepare_data(df)
    X, y = data_processor.prepare_features(df)
    X_scaled = data_processor.scale_features(X)
    
    print("\nTraining models...")
    if args.retrain or not os.path.exists(config.MODELS_DIR):
        print("Retraining models as requested...")
        trained_models = models.train_models(X, y, X_scaled)
    else:
        print("Loading saved models...")
        trained_models = models.load_saved_models()
    
    print("\nGenerating ensemble predictions...")
    # Generate predictions for each model
    predictions = {}
    for name, model in trained_models.items():
        if hasattr(model, 'predict_proba'):
            predictions[name] = model.predict_proba(X)[:, 1]
        else:
            # For LSTM/Keras models
            if name == 'lstm_attention':
                X_lstm = data_processor.prepare_lstm_data(df)
                pred = model.predict(X_lstm)
                # Flatten if needed
                if pred.ndim > 1:
                    pred = pred.flatten()
                # Pad the beginning with zeros to match df length
                pad_len = len(df) - len(pred)
                if pad_len > 0:
                    pred = np.pad(pred, (pad_len, 0), mode='constant', constant_values=0)
                predictions[name] = pred
            else:
                predictions[name] = model.predict(X)
    
    # Combine predictions using ensemble weights
    df['ensemble_signal'] = 0
    df['signal_strength'] = 0
    
    for name, pred in predictions.items():
        weight = config.ENSEMBLE_WEIGHTS[name]
        df['ensemble_signal'] += pred * weight
        df['signal_strength'] += np.abs(pred - 0.5) * weight
    
    # Store the raw ensemble prediction score for backtester
    df['prediction'] = df['ensemble_signal']
    # Convert to binary signals
    df['ensemble_signal'] = np.where(df['ensemble_signal'] > 0.5, 1, -1)
    # Add 'signal' column for backtester compatibility
    df['signal'] = df['ensemble_signal']
    
    print("\nRunning backtest...")
    backtest = backtester.Backtester(initial_cash=config.INITIAL_CASH)
    results = backtest.run(df)
    
    print("\nBacktest Results:")
    print(f"Initial Portfolio Value: ₹{results['initial_portfolio_value']:,.2f}")
    print(f"Final Portfolio Value: ₹{results['final_portfolio_value']:,.2f}")
    print(f"Total Return: {results['total_return']:.2%}")
    print(f"Annualized Return: {results['annualized_return']:.2%}")
    print(f"Sharpe Ratio: {results['sharpe_ratio']:.2f}")
    print(f"Maximum Drawdown: {results['max_drawdown']:.2%}")
    print(f"Number of Trades: {results['num_trades']}")
    
    print("\nSystem Summary:")
    print(f"Number of Models: {len(trained_models)}")
    print(f"Features Used: {len(data_processor.feature_cols)}")
    print(f"Data Period: {df.index[0]} to {df.index[-1]}")
    print(f"Final Sharpe Ratio: {results['sharpe_ratio']:.2f}")
    print(f"Final Return: {results['total_return']:.2%}")
    
    # Plot results
    visualizer.plot_results(backtest.portfolio_values, None)

if __name__ == "__main__":
    main()
