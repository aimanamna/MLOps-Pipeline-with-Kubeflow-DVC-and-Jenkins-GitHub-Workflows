1. Project Overview

This project demonstrates a complete end-to-end MLOps workflow using modern tools such as DVC, Minikube, Kubeflow Pipelines, Docker, Git/GitHub, and Jenkins/GitHub Actions CI.
The goal is to establish a reproducible, automated, and scalable machine learning pipeline for the Boston Housing dataset, which is versioned using DVC and processed inside Kubeflow components.

The pipeline consists of four modular components:

Data Extraction (via DVC) – Retrieves a versioned dataset from a DVC remote or tracked Git repository.

Data Preprocessing – Cleans the dataset, performs feature scaling, and generates train/test splits.

Model Training – Trains a RandomForestRegressor on the processed data and saves the model artifact.

Model Evaluation – Evaluates the model and logs performance metrics (MSE, RMSE, R²).

This setup showcases:

Reproducibility through dataset versioning (DVC).

Orchestration through Kubeflow Pipelines on Minikube.

Automation via CI tools (GitHub Actions/Jenkins).

Containerization and portability via Docker images.

The repository and pipeline form a complete MLOps system suitable for real-world production-grade machine learning workflows.

2. Setup Instructions

This section explains how to install and configure the required tools:
✔ Minikube
✔ Kubeflow Pipelines
✔ DVC data versioning
✔ GitHub repository structure

Follow each step to reproduce the environment.

2.1 Prerequisites

Ensure the following tools are installed on your machine:

Python 3.9+

Git

Docker

Kubectl

Minikube

DVC

Make (optional)

2.2 Setting Up DVC and Remote Storage
Step 1 — Initialize DVC in the Repository
dvc init
git add .dvc .dvcignore
git commit -m "Initialize DVC"

Step 2 — Add Dataset

Place the dataset inside:

data/raw_data.csv


Add it to DVC:

dvc add data/raw_data.csv
git add data/raw_data.csv.dvc data/.gitignore
git commit -m "Add dataset to DVC tracking"

Step 3 — Configure DVC Remote

Choose one option:

Option A: Local Folder Remote (simple for assignment)
mkdir -p /tmp/dvc_remote
dvc remote add -d myremote /tmp/dvc_remote
git commit -am "Configured local DVC remote"

Option B: Google Drive Remote
dvc remote add -d gdrive gdrive://<folder_id>

Option C: AWS S3 Remote
dvc remote add -d s3 s3://mybucket/dvc-storage

Step 4 — Push Dataset
dvc push
dvc status


Now the dataset is version-controlled, reproducible, and stored remotely.

2.3 Installing Minikube
Step 1 — Start Minikube
minikube start --driver=docker --cpus=4 --memory=8192

Step 2 — Check Status
minikube status

2.4 Installing Kubeflow Pipelines (Standalone Version)
Install KFP via Kubectl (Recommended for Assignments)

Run:

kubectl apply -k "github.com/kubeflow/pipelines/manifests/kustomize/cluster-install?ref=release-2.2"


Wait 2–3 minutes until all pods are running:

kubectl get pods -n kubeflow

Access KFP Dashboard
kubectl port-forward -n kubeflow svc/ml-pipeline-ui 8080:80


Open in browser:

http://localhost:8080

3. Pipeline Walkthrough

This section explains how the Kubeflow pipeline is created, compiled, and executed.

3.1 Compile Kubeflow Components

Run:

python src/pipeline_components.py


This generates:

components/
 ├── dvc_get_dataset.yaml
 ├── preprocess.yaml
 ├── train_model.yaml
 └── evaluate_model.yaml

3.2 Compile the Main Pipeline

Run:

python pipeline.py


This produces:

pipeline.yaml


The file contains the full pipeline graph.

3.3 Upload and Run the Pipeline in Kubeflow UI
Step 1 — Open Kubeflow UI

Go to:

http://localhost:8080

Step 2 — Upload Pipeline

Click Pipelines

Click Upload Pipeline

Select pipeline.yaml

Step 3 — Create an Experiment

Click:

Run Pipeline → Choose Experiment → Create New → Start

Step 4 — Provide Parameters

Example:

dvc_repo_url = https://github.com/<your-user>/mlops-kubeflow-assignment

Step 5 — Run the Pipeline

You will see all four components:

Data Extraction

Preprocessing

Model Training

Model Evaluation

All components should turn green (Succeeded).

3.4 View Outputs

You can inspect:

logs

model artifacts

evaluation metrics (MSE, RMSE, R²)

execution graph

This confirms that the pipeline executed successfully and produced reproducible outputs.

4. Conclusion

This repository implements a fully reproducible and automated MLOps workflow.
Using DVC, Kubeflow Pipelines, Minikube, Docker, and CI tools, it demonstrates:

Robust dataset versioning

Modular component-based ML pipelines

Automated pipeline execution and validation

Containerized, reproducible ML experiments

Clear separation of concerns between data, code, pipeline logic, and automation

The workflow can be expanded with:

Model monitoring

Model registry integration

Hyperparameter tuning

Kubernetes-native model deployment

This submission fulfills all task requirements for the final MLOps integration and documentation process.
