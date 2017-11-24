@echo off

rem # Kill the task/process of any previous Bokeh server:
taskkill /IM bokeh.exe /F

rem # Start the new Bokeh server by first finding the current IP:
for /f "tokens=1-2 delims=:" %%a in ('ipconfig^|find "IPv4"') do set ip=%%b
set ip=%ip:~1%

echo This batch file starts a Bokeh server that can be accessed remotely.
echo - For local access type "localhost/dataexplorer" in your browser.
echo - For remote access type "%IP%/dataexplorer" in your browser.

bokeh serve dataexplorer.py --show --allow-websocket-origin localhost:80 ^
 --allow-websocket-origin %ip%:80 --port 80 ^
 --log-level info

rem # 'log-level' can be one of: trace, debug, info, warning, error or critical
