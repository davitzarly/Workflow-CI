# Workflow CI - MLflow Project

Repository ini berisi MLflow Project dan GitHub Actions workflow untuk CI/CD.

## Struktur

```
Workflow-CI/
├── .github/
│   └── workflows/
│       └── mlflow_ci.yml
└── MLProject/
    ├── MLproject
    ├── conda.yaml
    ├── modelling.py
    └── predictive_maintenance_preprocessing/
```

## Setup

### 1. Setup Secrets di GitHub

Tambahkan secrets berikut di GitHub repository settings:

- `DOCKER_USERNAME`: Username Docker Hub
- `DOCKER_PASSWORD`: Password Docker Hub
- `MLFLOW_TRACKING_URI`: URI MLflow tracking server (opsional)

### 2. Menjalankan Workflow

Workflow akan otomatis berjalan ketika:
- Push ke branch main/master
- Pull request ke main/master
- Manual trigger via workflow_dispatch

### 3. MLflow Project Lokal

Untuk menjalankan MLflow project secara lokal:

```bash
cd MLProject
mlflow run . --experiment-name "Predictive Maintenance CI"
```

## Workflow Steps

1. Set up job
2. Checkout code
3. Set up Python 3.12.7
4. Check environment
5. Install dependencies
6. Set MLflow Tracking URI
7. Run MLflow project
8. Get latest MLflow run_id
9. Install Python dependencies
10. Upload artifacts (Google Drive/GitHub)
11. Build Docker image (Advanced)
12. Login to Docker Hub (Advanced)
13. Tag Docker image (Advanced)
14. Push Docker image (Advanced)






