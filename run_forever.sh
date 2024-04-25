#!/bin/bash

if [ -z "$1" ]; then
    echo "Usage: ./run_forever.sh <log_filename>"
    exit 1
fi

./run.sh >> "$1"

sleep_interval=10
while true
do
    if ! pgrep -f "run.sh" > /dev/null
    then
        echo "run.sh is stopped. Restarting in $sleep_interval seconds ..."
        sleep $sleep_interval
        sleep_interval=$((sleep_interval+sleep_interval))
        ./run.sh >> "$1"
    else
      sleep 10
    fi
done
