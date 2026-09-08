# ============================================================================
# SEL EKSPOR ARTEFAK MODEL  ->  jalankan di notebook AI_Detection.ipynb
# ----------------------------------------------------------------------------
# Prasyarat (sudah ada di notebook Anda):
#   - Xdf, y            (Tahap 5 - Matriks Fitur)
#   - build_preprocessor(svd_k, pca_k)   (Tahap 5)
#   - make_models()     (Tahap 6)
#
# Cukup dijalankan SEKALI. Hasilnya: model_detektor_catboost_469.joblib
# Lalu salin file itu ke  backend/artifacts/  pada proyek aplikasi.
# ============================================================================
import joblib
import sklearn
import catboost
from sklearn.pipeline import Pipeline

BEST_SVD, BEST_PCA = 300, 150   # konfigurasi terbaik (469 dim) sesuai jurnal

final_pipe = Pipeline([
    ("pre", build_preprocessor(BEST_SVD, BEST_PCA)),
    ("clf", make_models()["CatBoost"]),
])

# latih pada SELURUH data internal
final_pipe.fit(Xdf, y)

joblib.dump(final_pipe, "model_detektor_catboost_469.joblib")
print("OK -> model_detektor_catboost_469.joblib tersimpan.")
print("Catat versi library ini dan samakan di backend/requirements.txt:")
print("   scikit-learn =", sklearn.__version__)
print("   catboost     =", catboost.__version__)
