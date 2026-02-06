#!/usr/bin/env python3
"""
Neural Forex Trading App Setup Script
====================================

Professional setup script for the neural forex trading application.
Handles installation, configuration, and initial setup.

Usage:
    python setup.py --help
    python setup.py install
    python setup.py configure
    python setup.py test
"""

import sys
import os
import subprocess
import argparse
from pathlib import Path
import json
import yaml

def check_python_version():
    """Check if Python version is compatible"""
    if sys.version_info < (3, 8):
        print("❌ Error: Python 3.8+ required. Current version:", sys.version)
        return False
    print(f"✅ Python version: {sys.version}")
    return True

def install_requirements():
    """Install Python dependencies"""
    print("📦 Installing Python dependencies...")
    
    requirements_file = Path("requirements.txt")
    if not requirements_file.exists():
        print("❌ Error: requirements.txt not found")
        return False
    
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", str(requirements_file)])
        print("✅ Dependencies installed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Error installing dependencies: {e}")
        return False

def create_directories():
    """Create necessary directories"""
    print("📁 Creating directories...")
    
    directories = [
        "logs",
        "config",
        "models",
        "tests",
        "data",
        "backups"
    ]
    
    for directory in directories:
        dir_path = Path(directory)
        dir_path.mkdir(exist_ok=True)
        print(f"  📁 {directory}/")
    
    print("✅ Directories created")
    return True

def create_default_config():
    """Create default configuration files"""
    print("⚙️ Creating default configuration...")
    
    config_dir = Path("config")
    config_dir.mkdir(exist_ok=True)
    
    # Default application configuration
    app_config = {
        'app_name': 'Neural Forex Trading App',
        'version': '1.0.0',
        'update_interval': 5,
        'max_log_files': 10,
        'log_level': 'INFO',
        'auto_save_interval': 300,
        'theme': 'default',
        'window': {
            'width': 1000,
            'height': 700,
            'resizable': True,
            'center_on_screen': True
        }
    }
    
    # Default user configuration (auto-detect MT5 credentials)
    user_config = {
        'mt5': {
            'server': 'auto',  # Auto-detect from MT5
            'login': 'auto',   # Auto-detect from MT5
            'password': 'auto', # Use MT5 saved password
            'auto_connect': False,
            'connection_timeout': 30,
            'retry_attempts': 3,
            'auto_detect_credentials': True
        },
        'interface': {
            'show_tooltips': True,
            'auto_refresh_logs': True,
            'confirm_trades': True,
            'sound_notifications': False,
            'show_performance_charts': True
        },
        'notifications': {
            'trade_executions': True,
            'errors': True,
            'warnings': False,
            'system_updates': False,
            'email_notifications': False,
            'email_address': ''
        }
    }
    
    # Default trading configuration
    trading_config = {
        'general': {
            'trading_mode': 'demo',
            'default_risk_per_trade': 1.5,
            'default_confidence_threshold': 65,
            'max_concurrent_positions': 5,
            'max_daily_trades': 50,
            'max_daily_loss': 5.0,
            'auto_trading_enabled': False
        },
        'risk_management': {
            'enable_stop_loss': True,
            'enable_take_profit': True,
            'enable_trailing_stop': False,
            'enable_breakeven': False,
            'position_sizing_method': 'fixed_risk',
            'correlation_filter': True,
            'max_correlation': 0.7
        },
        'neural_network': {
            'model_path': 'models/neural_model.pth',
            'auto_load_model': True,
            'confidence_threshold': 65,
            'min_trades_for_retrain': 100,
            'retrain_interval_days': 30
        }
    }
    
    # Save configuration files
    config_files = {
        'app_config.yaml': app_config,
        'user_config.yaml': user_config,
        'trading_config.yaml': trading_config
    }
    
    for filename, config in config_files.items():
        config_file = config_dir / filename
        try:
            with open(config_file, 'w', encoding='utf-8') as f:
                yaml.dump(config, f, default_flow_style=False, indent=2, allow_unicode=True)
            print(f"  ⚙️ {filename}")
        except Exception as e:
            print(f"  ❌ Error creating {filename}: {e}")
            return False
    
    print("✅ Configuration files created")
    return True

def check_mt5_installation():
    """Check if MetaTrader 5 is installed"""
    print("🔍 Checking MetaTrader 5 installation...")
    
    # Common MT5 installation paths on Windows
    mt5_paths = [
        Path("C:/Program Files/MetaTrader 5/terminal64.exe"),
        Path("C:/Program Files (x86)/MetaTrader 5/terminal64.exe"),
        Path(os.path.expanduser("~/AppData/Roaming/MetaQuotes/Terminal/terminal64.exe"))
    ]
    
    for path in mt5_paths:
        if path.exists():
            print(f"✅ MetaTrader 5 found: {path}")
            return True, str(path)
    
    print("⚠️  MetaTrader 5 not found in common locations")
    print("   Please install MetaTrader 5 from: https://www.metatrader5.com/")
    return False, None

def test_installation():
    """Test the installation"""
    print("🧪 Testing installation...")
    
    # Test imports
    try:
        import tkinter
        print("  ✅ tkinter")
    except ImportError:
        print("  ❌ tkinter - Please install tkinter")
        return False
    
    try:
        import MetaTrader5
        print("  ✅ MetaTrader5")
    except ImportError:
        print("  ❌ MetaTrader5 - Run: pip install MetaTrader5")
        return False
    
    try:
        import torch
        print("  ✅ PyTorch")
    except ImportError:
        print("  ❌ PyTorch - Run: pip install torch")
        return False
    
    try:
        import yaml
        print("  ✅ PyYAML")
    except ImportError:
        print("  ❌ PyYAML - Run: pip install PyYAML")
        return False
    
    # Test app modules
    app_modules = [
        "app.config_manager",
        "app.model_manager", 
        "app.mt5_connector",
        "app.trading_engine"
    ]
    
    for module in app_modules:
        try:
            __import__(module)
            print(f"  ✅ {module}")
        except ImportError as e:
            print(f"  ❌ {module} - {e}")
            return False
    
    print("✅ Installation test passed")
    return True

def show_next_steps():
    """Show next steps for the user"""
    print("\n🎯 Next Steps:")
    print("1. Install MetaTrader 5 (if not already installed)")
    print("2. Create a demo trading account")
    print("3. Edit config/user_config.yaml with your MT5 credentials")
    print("4. Run: python main_app.py")
    print("5. Connect to MT5 and start with demo trading")
    
    print("\n⚠️  Important Safety Notes:")
    print("- Always start with demo account")
    print("- Set auto_trading_enabled: false initially")
    print("- Use small position sizes for testing")
    print("- Monitor performance before going live")
    
    print("\n📚 Documentation:")
    print("- README.md - Main documentation")
    print("- DEPLOYMENT_GUIDE.md - Detailed setup guide")
    print("- config/ - Configuration files")

def main():
    """Main setup function"""
    parser = argparse.ArgumentParser(description="Neural Forex Trading App Setup")
    parser.add_argument("--install", action="store_true", help="Install dependencies")
    parser.add_argument("--configure", action="store_true", help="Create configuration")
    parser.add_argument("--test", action="store_true", help="Test installation")
    parser.add_argument("--all", action="store_true", help="Run all steps")
    
    args = parser.parse_args()
    
    if not args.install and not args.configure and not args.test and not args.all:
        parser.print_help()
        return
    
    print("🚀 Neural Forex Trading App Setup")
    print("=" * 50)
    
    success = True
    
    # Check Python version
    if not check_python_version():
        return
    
    if args.install or args.all:
        if not install_requirements():
            success = False
    
    if args.configure or args.all:
        if not create_directories():
            success = False
        if not create_default_config():
            success = False
    
    if args.test or args.all:
        if not test_installation():
            success = False
    
    # Check MT5 installation
    mt5_installed, mt5_path = check_mt5_installation()
    
    if success:
        print("\n✅ Setup completed successfully!")
        show_next_steps()
    else:
        print("\n❌ Setup completed with errors. Please review the messages above.")

if __name__ == "__main__":
    main()
