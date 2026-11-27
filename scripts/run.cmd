@echo off
rem Windows task runner wrapper: scripts\run.cmd <task> [-Config <path>]  (see scripts\run.ps1)
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0run.ps1" %*
exit /b %ERRORLEVEL%
