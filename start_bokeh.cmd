@echo Welcome! This batch file finds your IP and starts a Bokeh server that can be accessed remotely.

for /f "tokens=1-2 delims=:" %%a in ('ipconfig^|find "IPv4"') do set ip=%%b
set ip=%ip:~1%

bokeh serve dataexplorer.py --show --allow-websocket-origin localhost:80 --allow-websocket-origin %ip%:80 --port 80

pause