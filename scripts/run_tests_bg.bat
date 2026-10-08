cd C:\BOT-TRADING
if not exist logs mkdir logs
C:\BOT-TRADING\.venv\Scripts\python.exe -m pytest tests/ deriv/tests/ -q --tb=short --no-header > logs\tests_final.txt 2>&1