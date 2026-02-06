#!/usr/bin/env python3
"""
Neural Forex Trading App
========================

Professional neural network-powered forex trading application.
Automatically connects to MT5, manages neural models, and executes trades.

Features:
- Professional GUI interface
- Automatic MT5 connection verification
- Neural network model management
- Real-time trading with confidence monitoring
- Professional logging and error handling
- GitHub-ready codebase

Author: Neural Trading System
Version: 1.0.0
"""

import sys
import os
import tkinter as tk
from tkinter import ttk, messagebox, scrolledtext
import threading
import time
import logging
from pathlib import Path
from datetime import datetime
import json

# Add app modules to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'app'))

from app.trading_engine import TradingEngine
from app.model_manager import NeuralModelManager
from app.mt5_connector import MT5Connector
from app.config_manager import ConfigManager

class NeuralTradingApp:
    """Professional neural forex trading application"""
    
    def __init__(self, root):
        self.root = root
        self.root.title("Neural Forex Trading App v1.0")
        self.root.geometry("800x600")
        self.root.resizable(True, True)
        
        # Initialize components
        self.config_manager = ConfigManager()
        self.model_manager = NeuralModelManager()
        self.mt5_connector = MT5Connector()
        self.trading_engine = None
        
        # App state
        self.is_trading = False
        self.model_loaded = False
        self.mt5_connected = False
        
        # Setup logging
        self.setup_logging()
        
        # Create GUI
        self.create_gui()
        
        # Check initial status
        self.check_initial_status()
    
    def setup_logging(self):
        """Setup professional logging"""
        # Create logs directory
        logs_dir = Path("logs")
        logs_dir.mkdir(exist_ok=True)
        
        # Configure logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(logs_dir / 'trading_app.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
        self.logger.info("Neural Trading App initialized")
    
    def create_gui(self):
        """Create professional GUI interface"""
        # Create notebook for tabs
        notebook = ttk.Notebook(self.root)
        notebook.pack(fill='both', expand=True, padx=10, pady=10)
        
        # Tab 1: Dashboard
        self.create_dashboard_tab(notebook)
        
        # Tab 2: Model Management
        self.create_model_tab(notebook)
        
        # Tab 3: Trading Control
        self.create_trading_tab(notebook)
        
        # Tab 4: Logs
        self.create_logs_tab(notebook)
        
        # Tab 5: Settings
        self.create_settings_tab(notebook)
    
    def create_dashboard_tab(self, notebook):
        """Create main dashboard tab"""
        dashboard_frame = ttk.Frame(notebook)
        notebook.add(dashboard_frame, text="Dashboard")
        
        # Status section
        status_frame = ttk.LabelFrame(dashboard_frame, text="System Status", padding=10)
        status_frame.pack(fill='x', padx=5, pady=5)
        
        # Status indicators
        self.status_labels = {}
        
        # MT5 Connection
        mt5_frame = ttk.Frame(status_frame)
        mt5_frame.pack(fill='x', pady=2)
        ttk.Label(mt5_frame, text="MT5 Connection:").pack(side='left')
        self.status_labels['mt5'] = ttk.Label(mt5_frame, text="❌ Disconnected", foreground='red')
        self.status_labels['mt5'].pack(side='left', padx=(10, 0))
        
        # Model Status
        model_frame = ttk.Frame(status_frame)
        model_frame.pack(fill='x', pady=2)
        ttk.Label(model_frame, text="Neural Model:").pack(side='left')
        self.status_labels['model'] = ttk.Label(model_frame, text="❌ Not Loaded", foreground='red')
        self.status_labels['model'].pack(side='left', padx=(10, 0))
        
        # Trading Status
        trading_frame = ttk.Frame(status_frame)
        trading_frame.pack(fill='x', pady=2)
        ttk.Label(trading_frame, text="Trading Engine:").pack(side='left')
        self.status_labels['trading'] = ttk.Label(trading_frame, text="❌ Stopped", foreground='red')
        self.status_labels['trading'].pack(side='left', padx=(10, 0))
        
        # Account Info
        account_frame = ttk.LabelFrame(dashboard_frame, text="Account Information", padding=10)
        account_frame.pack(fill='x', padx=5, pady=5)
        
        self.account_labels = {}
        account_info = [
            ("Account", "account"),
            ("Balance", "balance"),
            ("Equity", "equity"),
            ("Margin", "margin"),
            ("Free Margin", "free_margin")
        ]
        
        for label_text, key in account_info:
            frame = ttk.Frame(account_frame)
            frame.pack(fill='x', pady=1)
            ttk.Label(frame, text=f"{label_text}:").pack(side='left')
            self.account_labels[key] = ttk.Label(frame, text="N/A")
            self.account_labels[key].pack(side='left', padx=(10, 0))
        
        # Performance section
        perf_frame = ttk.LabelFrame(dashboard_frame, text="Trading Performance", padding=10)
        perf_frame.pack(fill='both', expand=True, padx=5, pady=5)
        
        self.performance_labels = {}
        perf_info = [
            ("Win Rate", "win_rate"),
            ("Total Trades", "total_trades"),
            ("Winning Trades", "winning_trades"),
            ("Daily P&L", "daily_pnl"),
            ("Total P&L", "total_pnl")
        ]
        
        for label_text, key in perf_info:
            frame = ttk.Frame(perf_frame)
            frame.pack(fill='x', pady=2)
            ttk.Label(frame, text=f"{label_text}:").pack(side='left')
            self.performance_labels[key] = ttk.Label(frame, text="0")
            self.performance_labels[key].pack(side='left', padx=(10, 0))
        
        # Control buttons
        control_frame = ttk.Frame(dashboard_frame)
        control_frame.pack(fill='x', padx=5, pady=10)
        
        self.connect_btn = ttk.Button(control_frame, text="Connect MT5", command=self.connect_mt5)
        self.connect_btn.pack(side='left', padx=5)
        
        self.load_model_btn = ttk.Button(control_frame, text="Load Model", command=self.load_model)
        self.load_model_btn.pack(side='left', padx=5)
        
        self.start_trading_btn = ttk.Button(control_frame, text="Start Trading", command=self.start_trading)
        self.start_trading_btn.pack(side='left', padx=5)
        
        self.stop_trading_btn = ttk.Button(control_frame, text="Stop Trading", command=self.stop_trading, state='disabled')
        self.stop_trading_btn.pack(side='left', padx=5)
    
    def create_model_tab(self, notebook):
        """Create model management tab"""
        model_frame = ttk.Frame(notebook)
        notebook.add(model_frame, text="Model Manager")
        
        # Model status
        status_frame = ttk.LabelFrame(model_frame, text="Neural Model Status", padding=10)
        status_frame.pack(fill='x', padx=5, pady=5)
        
        self.model_status_text = scrolledtext.ScrolledText(status_frame, height=10, width=70)
        self.model_status_text.pack(fill='both', expand=True)
        
        # Model controls
        controls_frame = ttk.Frame(model_frame)
        controls_frame.pack(fill='x', padx=5, pady=5)
        
        ttk.Button(controls_frame, text="Train New Model", command=self.train_model).pack(side='left', padx=5)
        ttk.Button(controls_frame, text="Load Model", command=self.load_model).pack(side='left', padx=5)
        ttk.Button(controls_frame, text="Validate Model", command=self.validate_model).pack(side='left', padx=5)
        
        # Model info
        info_frame = ttk.LabelFrame(model_frame, text="Model Information", padding=10)
        info_frame.pack(fill='x', padx=5, pady=5)
        
        self.model_info_text = scrolledtext.ScrolledText(info_frame, height=8, width=70)
        self.model_info_text.pack(fill='both', expand=True)
    
    def create_trading_tab(self, notebook):
        """Create trading control tab"""
        trading_frame = ttk.Frame(notebook)
        notebook.add(trading_frame, text="Trading Control")
        
        # Trading settings
        settings_frame = ttk.LabelFrame(trading_frame, text="Trading Settings", padding=10)
        settings_frame.pack(fill='x', padx=5, pady=5)
        
        # Risk management
        risk_frame = ttk.Frame(settings_frame)
        risk_frame.pack(fill='x', pady=5)
        
        ttk.Label(risk_frame, text="Risk per Trade (%):").pack(side='left')
        self.risk_var = tk.StringVar(value="1.5")
        risk_spinbox = ttk.Spinbox(risk_frame, from_=0.1, to=10.0, increment=0.1, textvariable=self.risk_var, width=10)
        risk_spinbox.pack(side='left', padx=(10, 0))
        
        # Confidence threshold
        conf_frame = ttk.Frame(settings_frame)
        conf_frame.pack(fill='x', pady=5)
        
        ttk.Label(conf_frame, text="Confidence Threshold (%):").pack(side='left')
        self.confidence_var = tk.StringVar(value="65")
        conf_spinbox = ttk.Spinbox(conf_frame, from_=50, to=95, increment=5, textvariable=self.confidence_var, width=10)
        conf_spinbox.pack(side='left', padx=(10, 0))
        
        # Trading pairs
        pairs_frame = ttk.LabelFrame(trading_frame, text="Trading Pairs", padding=10)
        pairs_frame.pack(fill='both', expand=True, padx=5, pady=5)
        
        # Create checkboxes for major pairs
        self.pair_vars = {}
        major_pairs = [
            "EURUSD", "GBPUSD", "USDJPY", "AUDUSD", 
            "USDCAD", "NZDUSD", "EURJPY", "GBPJPY"
        ]
        
        pairs_left = ttk.Frame(pairs_frame)
        pairs_left.pack(side='left', fill='both', expand=True)
        
        pairs_right = ttk.Frame(pairs_frame)
        pairs_right.pack(side='left', fill='both', expand=True)
        
        for i, pair in enumerate(major_pairs):
            var = tk.BooleanVar(value=True)
            self.pair_vars[pair] = var
            
            if i < len(major_pairs) // 2:
                ttk.Checkbutton(pairs_left, text=pair, variable=var).pack(anchor='w', pady=2)
            else:
                ttk.Checkbutton(pairs_right, text=pair, variable=var).pack(anchor='w', pady=2)
        
        # Active signals
        signals_frame = ttk.LabelFrame(trading_frame, text="Active Signals", padding=10)
        signals_frame.pack(fill='x', padx=5, pady=5)
        
        self.signals_tree = ttk.Treeview(signals_frame, columns=('Time', 'Pair', 'Action', 'Confidence', 'Price'), show='headings', height=8)
        self.signals_tree.heading('Time', text='Time')
        self.signals_tree.heading('Pair', text='Pair')
        self.signals_tree.heading('Action', text='Action')
        self.signals_tree.heading('Confidence', text='Confidence')
        self.signals_tree.heading('Price', text='Price')
        
        self.signals_tree.pack(fill='both', expand=True)
    
    def create_logs_tab(self, notebook):
        """Create logs tab"""
        logs_frame = ttk.Frame(notebook)
        notebook.add(logs_frame, text="Logs")
        
        # Log controls
        controls_frame = ttk.Frame(logs_frame)
        controls_frame.pack(fill='x', padx=5, pady=5)
        
        ttk.Button(controls_frame, text="Clear Logs", command=self.clear_logs).pack(side='left', padx=5)
        ttk.Button(controls_frame, text="Export Logs", command=self.export_logs).pack(side='left', padx=5)
        ttk.Button(controls_frame, text="Refresh", command=self.refresh_logs).pack(side='left', padx=5)
        
        # Log display
        self.log_text = scrolledtext.ScrolledText(logs_frame, height=20, width=80)
        self.log_text.pack(fill='both', expand=True, padx=5, pady=5)
        
        # Load initial logs
        self.load_logs()
    
    def create_settings_tab(self, notebook):
        """Create settings tab"""
        settings_frame = ttk.Frame(notebook)
        notebook.add(settings_frame, text="Settings")
        
        # MT5 Settings
        mt5_frame = ttk.LabelFrame(settings_frame, text="MT5 Settings", padding=10)
        mt5_frame.pack(fill='x', padx=5, pady=5)
        
        ttk.Label(mt5_frame, text="Server:").pack(anchor='w')
        self.server_var = tk.StringVar(value="MetaQuotes-Demo")
        ttk.Entry(mt5_frame, textvariable=self.server_var, width=30).pack(fill='x', pady=2)
        
        ttk.Label(mt5_frame, text="Login:").pack(anchor='w')
        self.login_var = tk.StringVar(value="")
        ttk.Entry(mt5_frame, textvariable=self.login_var, width=30).pack(fill='x', pady=2)
        
        ttk.Label(mt5_frame, text="Password:").pack(anchor='w')
        self.password_var = tk.StringVar(value="")
        ttk.Entry(mt5_frame, textvariable=self.password_var, show='*', width=30).pack(fill='x', pady=2)
        
        # App Settings
        app_frame = ttk.LabelFrame(settings_frame, text="Application Settings", padding=10)
        app_frame.pack(fill='x', padx=5, pady=5)
        
        ttk.Label(app_frame, text="Update Interval (seconds):").pack(anchor='w')
        self.update_interval_var = tk.StringVar(value="5")
        ttk.Entry(app_frame, textvariable=self.update_interval_var, width=10).pack(anchor='w', pady=2)
        
        ttk.Label(app_frame, text="Max Concurrent Trades:").pack(anchor='w')
        self.max_trades_var = tk.StringVar(value="5")
        ttk.Entry(app_frame, textvariable=self.max_trades_var, width=10).pack(anchor='w', pady=2)
        
        # Save settings button
        ttk.Button(settings_frame, text="Save Settings", command=self.save_settings).pack(pady=10)
    
    def check_initial_status(self):
        """Check initial system status"""
        self.update_status()
        self.root.after(5000, self.check_initial_status)  # Update every 5 seconds
    
    def update_status(self):
        """Update system status display"""
        try:
            # Update MT5 status
            if self.mt5_connector.is_connected():
                self.status_labels['mt5'].config(text="✅ Connected", foreground='green')
                account_info = self.mt5_connector.get_account_info()
                if account_info:
                    self.account_labels['account'].config(text=str(account_info.get('login', 'N/A')))
                    self.account_labels['balance'].config(text=f"${account_info.get('balance', 0):.2f}")
                    self.account_labels['equity'].config(text=f"${account_info.get('equity', 0):.2f}")
                    self.account_labels['margin'].config(text=f"${account_info.get('margin', 0):.2f}")
                    self.account_labels['free_margin'].config(text=f"${account_info.get('margin_free', 0):.2f}")
            else:
                self.status_labels['mt5'].config(text="❌ Disconnected", foreground='red')
            
            # Update model status
            if self.model_manager.is_model_loaded():
                self.status_labels['model'].config(text="✅ Loaded", foreground='green')
            else:
                self.status_labels['model'].config(text="❌ Not Loaded", foreground='red')
            
            # Update trading status
            if self.is_trading:
                self.status_labels['trading'].config(text="✅ Active", foreground='green')
            else:
                self.status_labels['trading'].config(text="❌ Stopped", foreground='red')
                
        except Exception as e:
            self.logger.error(f"Error updating status: {e}")
    
    def connect_mt5(self):
        """Connect to MT5"""
        def connect_thread():
            try:
                self.connect_btn.config(state='disabled', text="Connecting...")
                self.root.update()
                
                success = self.mt5_connector.connect()
                
                if success:
                    messagebox.showinfo("Success", "Connected to MT5 successfully!")
                    self.logger.info("MT5 connection established")
                else:
                    messagebox.showerror("Error", "Failed to connect to MT5")
                    self.logger.error("MT5 connection failed")
                
            except Exception as e:
                messagebox.showerror("Error", f"Connection error: {e}")
                self.logger.error(f"MT5 connection error: {e}")
            finally:
                self.connect_btn.config(state='normal', text="Connect MT5")
                self.update_status()
        
        threading.Thread(target=connect_thread, daemon=True).start()
    
    def load_model(self):
        """Load neural model"""
        def load_thread():
            try:
                self.load_model_btn.config(state='disabled', text="Loading...")
                self.root.update()
                
                success = self.model_manager.load_model()
                
                if success:
                    messagebox.showinfo("Success", "Neural model loaded successfully!")
                    self.logger.info("Neural model loaded")
                    self.display_model_info()
                else:
                    messagebox.showerror("Error", "Failed to load neural model")
                    self.logger.error("Model loading failed")
                
            except Exception as e:
                messagebox.showerror("Error", f"Model loading error: {e}")
                self.logger.error(f"Model loading error: {e}")
            finally:
                self.load_model_btn.config(state='normal', text="Load Model")
                self.update_status()
        
        threading.Thread(target=load_thread, daemon=True).start()
    
    def display_model_info(self):
        """Display model information"""
        try:
            info = self.model_manager.get_model_info()
            if info:
                self.model_info_text.delete(1.0, tk.END)
                self.model_info_text.insert(1.0, json.dumps(info, indent=2))
        except Exception as e:
            self.logger.error(f"Error displaying model info: {e}")
    
    def start_trading(self):
        """Start trading engine"""
        try:
            if not self.mt5_connector.is_connected():
                messagebox.showwarning("Warning", "Please connect to MT5 first!")
                return
            
            if not self.model_manager.is_model_loaded():
                messagebox.showwarning("Warning", "Please load neural model first!")
                return
            
            # Get settings
            risk_per_trade = float(self.risk_var.get()) / 100
            confidence_threshold = float(self.confidence_var.get()) / 100
            selected_pairs = [pair for pair, var in self.pair_vars.items() if var.get()]
            
            # Initialize trading engine
            self.trading_engine = TradingEngine(
                mt5_connector=self.mt5_connector,
                model_manager=self.model_manager,
                risk_per_trade=risk_per_trade,
                confidence_threshold=confidence_threshold,
                trading_pairs=selected_pairs
            )
            
            # Start trading
            self.trading_engine.start()
            self.is_trading = True
            
            self.start_trading_btn.config(state='disabled')
            self.stop_trading_btn.config(state='normal')
            
            messagebox.showinfo("Success", "Trading engine started!")
            self.logger.info("Trading engine started")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to start trading: {e}")
            self.logger.error(f"Trading startup error: {e}")
    
    def stop_trading(self):
        """Stop trading engine"""
        try:
            if self.trading_engine:
                self.trading_engine.stop()
            
            self.is_trading = False
            self.start_trading_btn.config(state='normal')
            self.stop_trading_btn.config(state='disabled')
            
            messagebox.showinfo("Info", "Trading engine stopped!")
            self.logger.info("Trading engine stopped")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to stop trading: {e}")
            self.logger.error(f"Trading shutdown error: {e}")
    
    def train_model(self):
        """Train new neural model"""
        # This would open a training dialog
        messagebox.showinfo("Info", "Model training feature coming soon!")
    
    def validate_model(self):
        """Validate loaded model"""
        # This would run model validation
        messagebox.showinfo("Info", "Model validation feature coming soon!")
    
    def clear_logs(self):
        """Clear log display"""
        self.log_text.delete(1.0, tk.END)
    
    def export_logs(self):
        """Export logs to file"""
        # Implementation for log export
        messagebox.showinfo("Info", "Log export feature coming soon!")
    
    def refresh_logs(self):
        """Refresh log display"""
        self.load_logs()
    
    def load_logs(self):
        """Load and display logs"""
        try:
            log_file = Path("logs/trading_app.log")
            if log_file.exists():
                with open(log_file, 'r') as f:
                    log_content = f.read()
                
                self.log_text.delete(1.0, tk.END)
                self.log_text.insert(1.0, log_content)
                
                # Auto-scroll to bottom
                self.log_text.see(tk.END)
        except Exception as e:
            self.logger.error(f"Error loading logs: {e}")
    
    def save_settings(self):
        """Save application settings"""
        try:
            settings = {
                'mt5_server': self.server_var.get(),
                'mt5_login': self.login_var.get(),
                'update_interval': int(self.update_interval_var.get()),
                'max_trades': int(self.max_trades_var.get())
            }
            
            self.config_manager.save_settings(settings)
            messagebox.showinfo("Success", "Settings saved successfully!")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to save settings: {e}")

def main():
    """Main application entry point"""
    try:
        # Create and run the application
        root = tk.Tk()
        app = NeuralTradingApp(root)
        
        # Start the GUI event loop
        root.mainloop()
        
    except Exception as e:
        print(f"Application error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
