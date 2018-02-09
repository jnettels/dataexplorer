#! /bin/sh

echo This shell script starts a Bokeh server on the Raspberry Pi
ip=$(hostname -I)  # Get the current IP address
ip=${ip%?}  # Remove the last character of the string
port=80  # Set a port (80 requires root rights)
loglevel="info"
N=3  # Number of worker processes
echo For remote access type $ip:$port/dataexplorer in remote browser
nohup /home/pi/berryconda3/bin/bokeh serve /home/pi/dataexplorer \
    --allow-websocket-origin $ip:$port --port $port --num-procs $N \
    --log-level $loglevel > /home/pi/dataexplorer/dataexplorer.log
