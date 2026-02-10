# 🚀 Installation Guide

This guide will walk you through setting up the Neural Forex Trading App step by step.

## 📋 What You Need

### Before You Start
- ✅ **Windows 10/11** computer
- ✅ **Administrator privileges** (to install software)
- ✅ **Internet connection** (for downloads)

### Software You'll Install
1. **Python 3.8+** (programming language)
2. **MetaTrader 5** (forex trading platform)
3. **Neural Forex Trading App** (this app!)

## 🔧 Step 1: Install Python

### Download Python
1. Go to [python.org](https://www.python.org/downloads/)
2. Click **"Download Python 3.11.x"** (or latest version)
3. Save the file to your **Downloads** folder

### Install Python
1. **Run** the downloaded file
2. ✅ **Important**: Check **"Add Python to PATH"**
3. Click **"Install Now"**
4. Wait for installation to complete
5. Click **"Close"**

### Verify Python Installation
1. Press **Windows Key + R**
2. Type `cmd` and press **Enter**
3. In the black window, type:
   ```
   python --version
   ```
4. You should see something like: `Python 3.11.x`

## 📊 Step 2: Install MetaTrader 5

### Download MT5
1. Go to [metatrader5.com](https://www.metatrader5.com/)
2. Click **"Download MT5"**
3. Save to **Downloads** folder

### Install MT5
1. **Run** the downloaded file
2. Follow the installation wizard
3. Choose **"Demo Account"** for testing (recommended)
4. Complete the installation

### Create Demo Account
1. **Open MT5** after installation
2. Click **"Open Demo Account"**
3. Fill in your details:
   - **Name**: Your name
   - **Email**: Your email
   - **Phone**: Your phone number
4. Choose **"MetaQuotes-Demo"** server
5. Create account with **$10,000** demo money
6. **Save login details** (you'll need them later)

## 🤖 Step 3: Install Neural Forex Trading App

### Download the App
1. Download all files from this repository
2. Extract to a folder like: `C:\NeuralForex\`

### Install Dependencies
1. Open **Command Prompt** (Windows Key + R, type `cmd`)
2. Navigate to the app folder:
   ```
   cd C:\NeuralForex\neural-forex-trading-app
   ```
3. Install required packages:
   ```
   pip install -r requirements.txt
   ```
4. Wait for installation (may take 5-10 minutes)

## 🎯 Step 4: First Launch

### Start the App
1. In the same command prompt, type:
   ```
   python main_app.py
   ```
2. **The app window should open!**

### Connect to MT5
1. **Click "Connect MT5"** button
2. The app should **auto-detect** your MT5 installation
3. **Login to your MT5** account manually first
4. **Return to the app** and click **"Connect MT5"** again

### Load Neural Model
1. **Click "Load Neural Model"**
2. Wait for the model to load (few seconds)
3. Status should show **"✅ Loaded"**

### Start Trading
1. **Click "Start Trading"**
2. The AI will begin making trading decisions!

## ✅ Success Checklist

If everything worked, you should see:
- ✅ **Neural Model**: Loaded (green)
- ✅ **MT5 Connection**: Connected (green)
- ✅ **Trading**: Active (green)
- ✅ **Predictions**: Appearing every few seconds

## 🆘 Troubleshooting

### Python Not Found
```
'python' is not recognized
```
**Solution**: 
1. Reinstall Python with **"Add to PATH"** checked
2. Restart command prompt

### MT5 Connection Failed
```
Failed to connect to MT5
```
**Solution**:
1. Make sure MT5 is **running and logged in**
2. **Restart both MT5 and the app**
3. Check that auto-trading is **enabled in MT5**

### No Neural Predictions
```
Model not loaded
```
**Solution**:
1. Click **"Load Neural Model"** first
2. Check that `neural_model.pth` file exists

### App Won't Start
```
Module not found
```
**Solution**:
1. Run: `pip install -r requirements.txt`
2. Check Python version: `python --version`

### Permission Errors
```
Access denied
```
**Solution**:
1. **Run command prompt as Administrator**
2. Right-click cmd → **"Run as administrator"**

## 📞 Need Help?

### Common Solutions
1. **Restart everything**: MT5, command prompt, and the app
2. **Check internet connection** for downloads
3. **Use Administrator mode** for installations
4. **Verify file paths** are correct

### Log Files
If something goes wrong, check:
- `logs/trading_app.log` - Main application logs
- `logs/trading_engine.log` - Trading operations

## 🎉 You're Ready!

Once everything is working:
- ✅ **Start with demo account** (no real money risk)
- ✅ **Monitor performance** for a few days
- ✅ **Learn the interface** before going live
- ✅ **Start with small amounts** when going live

**Happy AI-powered trading! 🚀📈🤖**

---

*Remember: Always test thoroughly with demo accounts before risking real money!*
