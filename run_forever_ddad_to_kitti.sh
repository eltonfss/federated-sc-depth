#!/bin/bash

if [ -z "$1" ]; then
    echo "Usage: ./run_forever_ddad_to_kitti.sh <log_filename>"
    exit 1
fi

./run_ddad_to_kitti.sh >> "$1"

sleep_interval=10
while true
do
    if ! pgrep -f "run_ddad_to_kitti.sh" > /dev/null
    then
        echo "run_ddad_to_kitti.sh is stopped. Restarting in $sleep_interval seconds ..."
        sleep $sleep_interval
        sleep_interval=$((sleep_interval+sleep_interval))
        ./run_ddad_to_kitti.sh >> "$1"
    else
      sleep 10
    fi
done