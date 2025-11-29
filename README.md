# MLOps-Pipeline-with-Kubeflow-DVC-and-Jenkins-GitHub-Workflows
To design, implement, and manage a complete Machine Learning Operations (MLOps) pipeline for a simple machine learning project. This assignment involves data versioning, pipeline orchestration, model training, and continuous integration using industry-standard tools like DVC, Kubeflow Pipelines, and Jenkins.
# MLOps Kubeflow Assignment

## Overview
This repository demonstrates an end-to-end MLOps workflow using DVC, Kubeflow Pipelines, and Minikube. The pipeline:
1. Fetches a DVC-tracked dataset (Boston housing)
2. Preprocesses (scaling, split)
3. Trains a RandomForest model
4. Evaluates and stores metrics

## Repo structure
- `data/` - data (raw/processed) (not committed)
- `src/` - pipeline components and training helper
- `components/` - compiled KFP component YAMLs
- `pipeline.py` - pipeline definition (compiles to pipeline.yaml)
- `pipeline.yaml` - compiled pipeline
- `requirements.txt`
- `Dockerfile`
- `Jenkinsfile`
- `.github/workflows/ci.yml`

## Setup
1. Install dependencies:

pip install -r requirements.txt

2. DVC:
dvc init
dvc remote add -d myremote <remote-path-or-url>

add your dataset

dvc add data/raw_data.csv
git add data/raw_data.csv.dvc .dvc .dvcignore
git commit -m "Add dataset to dvc"
dvc push

3. Compile components and pipeline:
python src/pipeline_components.py # creates components/*.yaml
python pipeline.py # creates pipeline.yaml

4. Run on Minikube + KFP:
- Start minikube, install Kubeflow Pipelines, open KFP UI, upload pipeline.yaml, run pipeline.

## CI
- Jenkinsfile provided for Jenkins pipeline.
- GitHub Actions workflow for CI (compiles pipeline and uploads artifacts).
