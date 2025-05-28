# AML

## Project Setup Guide

### Prerequisites

-   Docker (v20.10+ recommended)
-   Docker Compose (v2.0+ recommended)
-   NVIDIA Container Toolkit (for GPU support)
-   Python 3.12 (for local development)
-   Git+Git lfs
-   WSL (in the case of using Windows)

### Pull latest changes
```bash
git checkout main
git pull
git lfs fetch --all
git lfs pull
```

### Build the container
Dont be afraid if this process is slow - you are building an entire computer after all :).

```bash
sudo bash build.sh # or execute the content of build.sh
```

### Start the container
After running the script the first time a help message explaining how to proceed will be shown.

```bash
sudo bash run.sh # or execute the content of run.sh
```

## Windows notes
Ensure that you have WSL installed and execute these command using the WSL console (or git bash) as the project is build with unix like environment in mind. Additionaly Docker is sometimes misbehaving on windows but please make sure the Docker app is running when you execute the `build.sh` / `run.sh` commands. Also it might not be necessary to use `sudo` and `bash` when using WSL depending on how your environment is setup.

#### Possible problems:
1. Issues loading files as windows has issues with certain characters in file names. In those cases rename the problematic files accordingly.
2. Issues with zip (`build.sh: line 36: unzip: command not found`) in this case run `sudo apt install zip` in WSL
3. Problems finding files in WSL. Use ls/cd to navigate to the `aml` directory. Executing commands outside of it will not work.

## Running without CUDA GPU
To disable GPU support (and therefore avoid related issues) simply delete `docker-compose.override.yaml` from the root. However be careful not to commit these changes :).

## Project Structure

├── aml # Main program files (/aml in container) \
├── data # Dataset related files (/data in container) \
├── logs # Prograns logs (/logs in container) \
├── weights # Weights for models (/weights in container) \
├── API-docs.md # Docs for the API \
├── build.sh # Build script for Docker \
├── docker-compose.yaml # Docker orchestration config \
├── docker-compose.override.yaml # GPU support for Docker \
├── Dockerfile # Definition of Docker image \
├── .env # Project wide settings \
├── .env-example # Example for .env \
├── pyproject.toml # Development dependencies \
├── README.md # This file \
├── requirements.txt # Runtime dependencies \
├── .github # CI pipeline \
└── run.sh # Run script for Docker \