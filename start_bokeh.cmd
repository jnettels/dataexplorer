@echo off

echo Try to kill the task/process of any previous Bokeh server:
taskkill /IM bokeh.exe /F

rem # Find the current IP:
for /f "tokens=1-2 delims=:" %%a in ('ipconfig^|find "IPv4"') do set ip=%%b
set ip=%ip:~1%

rem # Read command line option #1 as port number (or set 80 to default):
set port=%1
if not defined port set port=80

rem # Read command line option #2 as log-level (default is 'critical'):
rem # 'log-level' can be one of: trace, debug, info, warning, error or critical
set loglevel=%2
if not defined loglevel set loglevel=critical

rem # The full path to the batch file's directory
set folder=%~dp0

echo This batch file starts a Bokeh server that can be accessed remotely.
echo - For local access type "localhost:%port%/dataexplorer" in local browser.
echo - For remote access type "%IP%:%port%/dataexplorer" in remote browser.

rem # Start Bokeh server with enabled remote access and the given log-level
bokeh serve %folder% --show --allow-websocket-origin localhost:%port% ^
 --allow-websocket-origin %ip%:%port% --port %port% ^
 --log-level %loglevel%

rem # How to add this script to the taskbar:
rem # Add a shortcut to cmd.exe to the taskbar, then modify its properties:
rem # - target: C:\WINDOWS\system32\cmd.exe /C start_bokeh.cmd 80 info
rem # - run in: "C:\Users\nettelstroth\Documents\07 Python\DataExplorer"
rem #
