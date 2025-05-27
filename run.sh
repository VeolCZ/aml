#!/bin/bash

# Copy this to run manualy
<<<<<<< HEAD
#docker compose run --rm --remove-orphans -p 8000:8000 app python3 main.py "$@"
docker compose run --rm --remove-orphans -p 8000:8000 -p 8080:8080 app python3 main.py "$@"
=======
docker compose run --rm --remove-orphans --service-ports app python3 main.py "$@"
>>>>>>> origin/idiot_proof
