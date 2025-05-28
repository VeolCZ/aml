#!/bin/bash

# Copy this to run manualy
docker compose run --rm --remove-orphans --service-ports app python3 main.py "$@"