# Start the app without writing .pyc files for this session
$env:PYTHONDONTWRITEBYTECODE = '1'
Write-Output "PYTHONDONTWRITEBYTECODE=$env:PYTHONDONTWRITEBYTECODE"

# Use the exact Python executable if you need to; otherwise rely on PATH
python run.py
