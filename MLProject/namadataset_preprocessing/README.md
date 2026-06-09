# namadataset_preprocessing

Folder ini berisi dataset hasil preprocessing untuk workflow CI:

- `train_preprocessed.csv`
- `test_preprocessed.csv`
- `metadata.json`
- `preprocessor.joblib`
- `feature_names.txt`

Workflow CI memakai data ini lebih dulu. Jika file belum tersedia, script CI akan mencoba memakai raw dataset `hotel_bookings.csv`.
