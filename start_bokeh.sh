#! /bin/sh

# This shell script starts a Bokeh server on the Raspberry Pi.
# For autostart, place the path to this script in /etc/rc.local

ip=$(hostname -I)  # Get the current IP address
ip=${ip%?}  # Remove the last character of the string
port=80  # Set a port (80 requires root rights)
loglevel="info"
N=3  # Number of worker processes
echo For remote access type $ip:$port/dataexplorer in remote browser
nohup /home/pi/berryconda3/bin/bokeh serve /home/pi/dataexplorer \
    --allow-websocket-origin $ip:$port \
    --allow-websocket-origin=dataexplorer.igs.bau.tu-bs.de:$port \
    --port $port --num-procs $N --log-level $loglevel \
    --websocket-max-message-size 100000000 \
    --args --server_mode \
    > /home/pi/dataexplorer/dataexplorer.log 2>&1 &
