# AML

### Docker Setup Guide

#### Prerequisites

-   Docker (v20.10+ recommended)
-   Docker Compose (v2.0+ recommended)
-   NVIDIA Container Toolkit (for GPU support)
-   Python 3.12 (for local development)

### Quick Start

#### 1. Configure Environment

```bash
cp .env.example .env # copy and edit the environment file
```

### Usage

#### Build the container

```bash
sudo bash build.sh # or execute the content of build.sh
```

#### Start the container

```bash
sudo bash run.sh # or execute the content of run.sh
```

### Contributions

First pull the repository to you environment using your preferd way. Afterwards and before making new brach execute the following.

```bash
git fetch
git checkout main
```

To create your own branch make sure that you are on the main branch and then create a branch starting off main.

```bash
git fetch # get new data from github
git branch # should show main
git checkout main # if main is not shown swith branch to main
git branch NAME # make a branch with a descriptive name
git checkout NAME # go to your branch
```

Afterwards develop on your branch like you would normally.

```bash
git add . # add your changes to the branch
git commit -m MESSAGE # commit the changes to the branch with descpritive message
git push # push your changes to github
```

Once you are satisfied with the state of your branch move to github. [Navigate to](https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/proposing-changes-to-your-work-with-pull-requests/creating-a-pull-request) `Pull requests` -> `New pull request`. Here select `base: main` and `compare: YOUR_BRACH`. Then create the pull request. Now you should see all the changes you have made in comparrison with the main branch. Verify that this is what you wanted. Also automatic checks will run to validate the code. Now you need to assign reviewers and hit others up on whatsapp. Once the pull request (PR) gets approved by others it will be merged to the main branch and you can make a new branch from the new main. If the changes are not mergable (you will see this) you need to perform a merge but in that case it is easier to hit someone up to help you resolve the merge.

### Project Structure

├── aml/ # Project code (→ /aml in container)\
├── data/ # Persistent data (→ /data)\
├── logs/ # Application logs (→ /logs)\
├── pyproject.toml # Development dependencies\
├── Dockerfile # Container definition\
├── docker-compose.yml # Orchestration config\
├── requirements.txt # Python dependencies\
└── .env # Environment config
