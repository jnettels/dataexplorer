#! /bin/sh

echo This shell script starts a Bokeh server on Linux systems
ip=$(hostname -I)
port=80
loglevel="info"
echo For remote access type $ip:$port/dataexplorer in remote browser
nohup bokeh serve ../dataexplorer --allow-websocket-origin $ip:$port \
    --port $port --log-level $loglevel > dataexplorer.log