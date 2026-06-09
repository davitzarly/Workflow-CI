# Workflow CI

Folder ini menyiapkan contoh repository untuk kriteria 3. Salin isi folder ini ke repository GitHub public khusus workflow CI, lalu aktifkan GitHub Actions.

Workflow advanced berada di:

```text
.github/workflows/mlflow-ci.yml
```

Secrets opsional:

- `DAGSHUB_REPO_OWNER`
- `DAGSHUB_REPO_NAME`
- `DAGSHUB_USER_TOKEN`
- `DOCKERHUB_USERNAME`
- `DOCKERHUB_TOKEN`

Jika Docker Hub secrets belum tersedia, step build/push Docker akan dilewati.
