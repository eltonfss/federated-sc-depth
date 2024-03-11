#!/bin/bash

if [ -z "$1" ]; then
    echo "Usage: ./run_forever.sh <log_filename>"
    exit 1
fi

./run_ddad_to_kitti.sh >> "$1"

while true
do
    if ! pgrep -f "run.sh" > /dev/null
    then
        echo "run.sh is stopped. Restarting in 10 seconds ..."
        sleep 10
        ./run.sh >> "$1"
    else
      sleep 10
    fi
done
