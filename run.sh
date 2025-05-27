#!/bin/bash

# Copy this to run manualy
#docker compose run --rm --remove-orphans -p 8000:8000 app python3 main.py "$@"
docker compose run --rm --remove-orphans -p 8000:8000 -p 8080:8080 app python3 main.py "$@"
