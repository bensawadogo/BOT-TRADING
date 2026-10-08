@echo off
cd /d C:\BOT-TRADING
if not exist logs mkdir logs
.venv\Scripts\python.exe -W ignore deriv\walk_forward_optimizer.py --symbol frxEURUSD --count 3000 > logs\wfo_eurusd.txt 2> logs\wfo_eurusd_err.txt
.venv\Scripts\python.exe -W ignore deriv\symbol_optimizer.py --count 900 > logs\symopt.txt 2> logs\symopt_err.txt
echo TERMINE > logs\validation_done.txt
