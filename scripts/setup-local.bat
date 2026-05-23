@echo off
REM Wrapper batch pour lancer setup-local.ps1 (utilisateurs qui ne sont pas en PowerShell)
REM Usage : .\scripts\setup-local.bat
powershell -ExecutionPolicy Bypass -File "%~dp0setup-local.ps1" %*
