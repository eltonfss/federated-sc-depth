#!/bin/bash

if [ -z "$1" ]; then
    echo "Usage: ./run_forever_kitti_to_ddad.sh <log_filename>"
    exit 1
fi

./run_kitti_to_ddad.sh >> "$1"

sleep_interval=10
while true
do
    if ! pgrep -f "run_kitti_to_ddad.sh" > /dev/null
    then
        echo "run_kitti_to_ddad.sh is stopped. Restarting in $sleep_interval seconds ..."
        sleep $sleep_interval
        sleep_interval=$((sleep_interval+sleep_interval))
        ./run_kitti_to_ddad.sh >> "$1"
    else
      sleep 10
    fi
done