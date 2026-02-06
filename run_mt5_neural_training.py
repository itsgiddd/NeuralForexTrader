#!/usr/bin/env python3
"""
MT5 Neural Training Setup and Execution Script
===========================================

This script sets up and executes the neural network training on real MT5 data.

Before running:
1. Ensure MetaTrader 5 is installed and running
2. MT5 must be logged into your trading account
3. Install required packages: pip install MetaTrader5 pandas numpy scikit-learn

Usage:
python run_mt5_neural_training.py

This will:
1. Connect to your MT5 terminal
2. Download historical data for major currency pairs
3. Train the neural network on real market data
4. Compare performance with the original rule-based system
5. Generate comprehensive performance reports
"""

import sys
import subprocess
import importlib
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def check_and_install_requirements():
    """Check and install required packages"""
    
    required_packages = [
        'MetaTrader5',
        'pandas',
        'numpy', 
        'torch',
        'scikit-learn',
        'scipy'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            importlib.import_module(package.replace('-', '_').lower())
            logger.info(f"✓ {package} is installed")
        except ImportError:
            missing_packages.append(package)
            logger.warning(f"✗ {package} is missing")
    
    if missing_packages:
        logger.info(f"Installing missing packages: {missing_packages}")
        for package in missing_packages:
            try:
                subprocess.check_call([sys.executable, '-m', 'pip', 'install', package])
                logger.info(f"✓ {package} installed successfully")
            except subprocess.CalledProcessError:
                logger.error(f"✗ Failed to install {package}")
                return False
    
    return True

def check_mt5_connection():
    """Check if MT5 is accessible"""
    
    try:
        import MetaTrader5 as mt5
        
        if not mt5.initialize():
            logger.error("Failed to initialize MT5 connection")
            logger.error("Please ensure:")
            logger.error("1. MetaTrader 5 is installed")
            logger.error("2. MT5 is running")
            logger.error("3. You are logged into your trading account")
            return False
        
        account_info = mt5.account_info()
        if account_info is None:
            logger.error("Failed to get account info - please check MT5 login")
            return False
        
        logger.info(f"✓ Connected to MT5 - Account: {account_info.login}")
        logger.info(f"  Server: {account_info.server}")
        logger.info(f"  Balance: ${account_info.balance:,.2f}")
        
        mt5.shutdown()
        return True
        
    except Exception as e:
        logger.error(f"MT5 connection error: {str(e)}")
        return False

def get_training_parameters():
    """Get training parameters from user or use defaults"""
    
    print("\n" + "="*60)
    print("MT5 NEURAL NETWORK TRAINING SETUP")
    print("="*60)
    
    # Default parameters
    default_params = {
        'training_start': '2022-01-01',
        'training_end': '2023-06-01', 
        'testing_start': '2023-06-01',
        'testing_end': '2023-12-31',
        'batch_size': 32,
        'num_epochs': 50,
        'learning_rate': 0.001,
        'ensemble_size': 3
    }
    
    print("Training Parameters (press Enter to use defaults):")
    print(f"Training Start Date: {default_params['training_start']}")
    print(f"Training End Date: {default_params['training_end']}")
    print(f"Testing Start Date: {default_params['testing_start']}")
    print(f"Testing End Date: {default_params['testing_end']}")
    print(f"Batch Size: {default_params['batch_size']}")
    print(f"Number of Epochs: {default_params['num_epochs']}")
    print(f"Learning Rate: {default_params['learning_rate']}")
    print(f"Ensemble Size: {default_params['ensemble_size']}")
    
    use_defaults = input("\nUse default parameters? (y/n): ").lower().strip()
    
    if use_defaults == 'y':
        return default_params
    
    # Custom parameters (simplified input)
    print("\nEnter custom parameters (or press Enter to skip):")
    
    params = default_params.copy()
    
    try:
        batch_size = input(f"Batch Size [{default_params['batch_size']}]: ").strip()
        if batch_size:
            params['batch_size'] = int(batch_size)
        
        epochs = input(f"Number of Epochs [{default_params['num_epochs']}]: ").strip()
        if epochs:
            params['num_epochs'] = int(epochs)
        
        lr = input(f"Learning Rate [{default_params['learning_rate']}]: ").strip()
        if lr:
            params['learning_rate'] = float(lr)
        
    except ValueError:
        logger.warning("Invalid input, using default parameters")
        return default_params
    
    return params

def run_training():
    """Execute the main training process"""
    
    print("\n" + "="*60)
    print("STARTING NEURAL NETWORK TRAINING")
    print("="*60)
    
    try:
        # Import our training system
        from mt5_neural_training_system import run_mt5_neural_training
        
        # Run the training
        run_mt5_neural_training()
        
        print("\n" + "="*60)
        print("TRAINING COMPLETED SUCCESSFULLY!")
        print("="*60)
        print("\nCheck the following files for results:")
        print("- mt5_neural_training_results.json (detailed results)")
        print("- models/ directory (trained neural network models)")
        print("- All intermediate training logs above")
        
    except Exception as e:
        logger.error(f"Training execution failed: {str(e)}")
        print(f"\nError: {str(e)}")
        print("\nTroubleshooting tips:")
        print("1. Ensure MT5 is running and logged in")
        print("2. Check that you have sufficient historical data")
        print("3. Verify all required packages are installed")
        print("4. Check the logs above for specific error messages")

def main():
    """Main execution function"""
    
    print("MT5 Neural Trading System Training")
    print("=" * 40)
    
    # Step 1: Check requirements
    print("\n1. Checking requirements...")
    if not check_and_install_requirements():
        print("Failed to install required packages. Please install manually:")
        print("pip install MetaTrader5 pandas numpy torch scikit-learn scipy")
        return
    
    # Step 2: Check MT5 connection
    print("\n2. Checking MT5 connection...")
    if not check_mt5_connection():
        print("\nCannot connect to MT5. Please ensure:")
        print("- MetaTrader 5 is installed and running")
        print("- You are logged into your trading account")
        print("- MT5 allows automated trading (Expert Advisors)")
        return
    
    # Step 3: Get parameters
    print("\n3. Configuring training parameters...")
    params = get_training_parameters()
    
    # Step 4: Run training
    print(f"\n4. Starting training with parameters:")
    for key, value in params.items():
        print(f"   {key}: {value}")
    
    input("\nPress Enter to start training...")
    
    run_training()

if __name__ == "__main__":
    main()
