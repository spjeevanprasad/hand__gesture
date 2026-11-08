$ErrorActionPreference = "Continue"
Set-Location -Path $PSScriptRoot

$pythonExe = "$env:LOCALAPPDATA\Programs\Python\Python313\python.exe"
if (Test-Path $pythonExe) {
    & $pythonExe main.py @args
} else {
    python main.py @args
}

Write-Host "`nPress Enter to exit..."
Read-Host
