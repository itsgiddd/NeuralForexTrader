# MBATS (Microservices Based Algorithmic Trading System)

MBATS is an algorithmic trading system that combines deterministic pattern recognition, safety gating, and a self-learning neural decision model to score trade setups.

> **Risk disclaimer:** Trading is risky and there are no guarantees of profit. This system is for educational and research use.

## Key Components

- **Market Context**: Trend, strength, volatility, session quality, and momentum scoring.
- **Pattern Recognition**: Flags, pennants, rectangles, triangles, wedges, head & shoulders, double tops/bottoms, rounding, diamonds, and teacup patterns.
- **Safety Rules**: Push exhaustion (>=4 pushes), minimum 1:2 risk-reward, and SL polarity validation.
- **Neural Decision Model**: Self-learning model that updates on trade outcomes to refine scoring over time.

## Platform Notes

- The MetaTrader5 Python API requires a **Windows** MT5 terminal.
- On macOS, you can run MT5 via a Windows VM or a dedicated Windows host and run the bot there.

## Running the Bot

1. Install MT5 and Python dependencies (see `mt5_windows_setup.md`).
2. Start the bot:
   ```bash
   python live_trader.py
   ```

## Project Goal

This project aims to systematize high-quality pattern trading. It does **not** promise or guarantee specific account growth.
