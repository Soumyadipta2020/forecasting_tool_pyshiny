# Remove all __pycache__ directories and .pyc files under the project

Write-Output "Removing __pycache__ directories..."
Get-ChildItem -Path . -Recurse -Directory -Filter "__pycache__" | ForEach-Object { Remove-Item $_.FullName -Recurse -Force }

Write-Output "Removing .pyc files..."
Get-ChildItem -Path . -Recurse -Include "*.pyc" | Remove-Item -Force -ErrorAction SilentlyContinue

Write-Output "Done."
