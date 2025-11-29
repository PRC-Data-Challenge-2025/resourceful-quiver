#! /usr/bin/bash
mc ls --recursive opensky/prc-2025-resourceful-quiver/ | grep '\.parquet$' | grep -oP '(?<=_v)\d+' | sort -n | uniq | tail -n 1