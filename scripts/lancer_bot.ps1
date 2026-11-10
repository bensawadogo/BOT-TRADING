# Lance le bot de tendance (démo) s'il ne tourne pas déjà. Appelé par la tâche planifiée
# « BOT-TRADING trend_bot » (à l'ouverture de session et tous les jours à 06:45 UTC, en
# réveillant le PC), ou à la main :  powershell -File scripts\lancer_bot.ps1
#
# Le bot écrit lui-même son journal dans logs\trend_bot.log (main.py) et empêche la veille
# automatique tant qu'il tourne (runner.empecher_la_veille). Un seul exemplaire à la fois.
$ErrorActionPreference = "Stop"
$racine = Split-Path -Parent $PSScriptRoot
Set-Location $racine
New-Item -ItemType Directory -Force -Path (Join-Path $racine "logs") | Out-Null
$log = Join-Path $racine "logs\trend_bot.log"
$horodatage = (Get-Date).ToUniversalTime().ToString("yyyy-MM-dd HH:mm:ss")

$deja = Get-CimInstance Win32_Process -Filter "Name='python.exe'" |
    Where-Object { $_.CommandLine -like "*main.py trend --live --loop*" }
if ($deja) {
    Add-Content -Encoding UTF8 $log "$horodatage lancer_bot : deja en cours (PID $($deja.ProcessId -join ',')), rien à faire"
    exit 0
}

$python = Join-Path $racine ".venv\Scripts\python.exe"
Add-Content -Encoding UTF8 $log "$horodatage lancer_bot : demarrage"
# stderr du processus (plantage avant la mise en place du journal) : fichier à part, écrasé à chaque lancement
Start-Process -FilePath $python -ArgumentList "main.py", "trend", "--live", "--loop" `
    -WorkingDirectory $racine -WindowStyle Hidden `
    -RedirectStandardError (Join-Path $racine "logs\trend_bot_stderr.log")
