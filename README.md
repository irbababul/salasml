# salasml

Library Python untuk mempercepat workflow Machine Learning — pipeline preprocessing, hyperparameter tuning, feature importance, dan utilitas lainnya menggunakan scikit-learn.

---

## Instalasi

### Cara 1 — Install dari GitHub (direkomendasikan)

```bash
pip install git+https://github.com/salasml/salasml.git
```

### Cara 2 — Install dari file `.whl`

Download file `salasml-1.0.0-py3-none-any.whl` dari [Releases](../../releases), lalu:

```bash
pip install salasml-1.0.0-py3-none-any.whl
```

### Cara 3 — Install dari source (clone repo)

```bash
git clone https://github.com/salasml/salasml.git
cd salasml
pip install -e .
```

### Dependencies

salasml membutuhkan package berikut (otomatis terinstall):

```
numpy
pandas
scikit-learn
matplotlib
seaborn
xgboost
scipy
```

Untuk Bayesian Search (opsional):

```bash
pip install scikit-optimize
```

---

## Penggunaan

### Import Utama

```python
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

from salasml.pipeline import num_pipe, cat_pipe
from salasml.utils import save_model, load_model
from salasml.plot import plot_missing_value
from salasml.feature_importance import mean_score_decrease
```

---

## Modul

### `salasml.pipeline`

Pipeline otomatis untuk preprocessing fitur numerik dan kategorikal.

#### `num_pipe()`

```python
from salasml.pipeline import num_pipe

num_pipe(
    impute='median',      # {'knn', 'mean', 'median', None}
    poly=None,            # int atau None — degree polynomial features
    transform=None,       # {'yeo-johnson', 'box-cox', None}
    scaling=None,         # {'standard', 'minmax', 'robust', 'maxabs', 'normalize', None}
    n_neighbors=5,        # dipakai jika impute='knn'
    weights='uniform'     # dipakai jika impute='knn'
)
```

Contoh:

```python
from sklearn.compose import ColumnTransformer

num_cols = ['age', 'income', 'spending']
cat_cols = ['gender', 'city']

preprocessor = ColumnTransformer([
    ('numeric',   num_pipe(scaling='standard'), num_cols),
    ('categoric', cat_pipe(encoder='onehot'),   cat_cols),
])
```

#### `cat_pipe()`

```python
from salasml.pipeline import cat_pipe

cat_pipe(
    impute='most_frequent',  # {'most_frequent', None}
    encoder='onehot'         # {'onehot', 'ordinal', None}
)
```

---

### `salasml.utils`

#### `save_model()`

Simpan model ke file `.pkl`.

```python
from salasml.utils import save_model

save_model(model, "my_model.pkl")               # simpan seluruh search object
save_model(model.best_estimator_, "best.pkl")   # simpan best estimator saja
# File tersimpan di folder 'model/my_model.pkl' (default)
```

#### `load_model()`

Load model dari file `.pkl`.

```python
from salasml.utils import load_model

model = load_model("model/my_model.pkl")
```

---

### `salasml.plot`

#### `plot_missing_value()`

Visualisasi missing value sebagai heatmap.

```python
from salasml.plot import plot_missing_value

plot_missing_value(df)

# Kembalikan DataFrame ringkasan
df_miss = plot_missing_value(df, return_df=True)

# Orientasi vertikal
plot_missing_value(df, feature_alignment='vertical')
```

Parameter:

| Parameter | Default | Opsi |
|---|---|---|
| `return_df` | `False` | `True` / `False` |
| `feature_alignment` | `'horizontal'` | `'horizontal'`, `'vertical'` |
| `figsize` | `(15, 8)` | tuple `(lebar, tinggi)` |

---

### `salasml.feature_importance`

#### `mean_score_decrease()`

Hitung permutation importance untuk setiap fitur.

```python
from salasml.feature_importance import mean_score_decrease

df_imp = mean_score_decrease(X_train, y_train, model, plot=True, topk=10)
```

Parameter:

| Parameter | Default | Keterangan |
|---|---|---|
| `plot` | `False` | Tampilkan barplot |
| `topk` | `None` | Tampilkan k fitur terpenting |
| `n_fold` | `5` | Jumlah permutasi |
| `normalize` | `False` | Normalisasi importance ke 0–1 |
| `random_state` | `42` | Random seed |

#### `mean_loss_decrease()`

Feature importance dari tree-based model (Decision Tree, Random Forest, XGBoost).

```python
from salasml.feature_importance import mean_loss_decrease

df_imp = mean_loss_decrease(X_train, model, plot=True, topk=10)
```

---

### `salasml.tuning`

Kumpulan param grid/dist siap pakai untuk GridSearchCV dan RandomizedSearchCV.

#### `grid_search_params` (gsp)

```python
from sklearn.model_selection import GridSearchCV
from salasml.tuning import grid_search_params as gsp

model = GridSearchCV(pipeline, gsp.knn_params, cv=5, scoring='r2', n_jobs=-1, verbose=1)
model.fit(X_train, y_train)

print(model.best_params_)
print(model.score(X_train, y_train), model.best_score_, model.score(X_test, y_test))
```

Param yang tersedia:

| Attribute | Model |
|---|---|
| `gsp.knn_params` | K-Nearest Neighbor |
| `gsp.svm_params` | Support Vector Machine |
| `gsp.rf_params` | Random Forest |
| `gsp.xgb_params` | XGBoost |
| `gsp.linreg_params` | Linear Regression |
| `gsp.enet_params` | ElasticNet |
| `gsp.logreg_params` | Logistic Regression |
| `gsp.knn_poly_params` | KNN + Polynomial Features |
| `gsp.rf_poly_params` | Random Forest + Polynomial Features |
| *(dan lainnya)* | |

#### `random_search_params` (rsp)

```python
from sklearn.model_selection import RandomizedSearchCV
from salasml.tuning import random_search_params as rsp

model = RandomizedSearchCV(pipeline, rsp.rf_params, cv=5, scoring='r2',
                           n_iter=50, n_jobs=-1, verbose=1, random_state=42)
model.fit(X_train, y_train)

print(model.best_params_)
print(model.score(X_train, y_train), model.best_score_, model.score(X_test, y_test))
```

Attribute sama dengan `gsp` di atas.

#### `bayes_search_params` (bsp) — membutuhkan `scikit-optimize`

```bash
pip install scikit-optimize
```

```python
from skopt import BayesSearchCV
from salasml.tuning import bayes_search_params as bsp

model = BayesSearchCV(pipeline, bsp.rf_params, cv=5, scoring='r2',
                      n_iter=50, n_jobs=-1, verbose=1, random_state=42)
model.fit(X_train, y_train)
```

---

## Contoh Lengkap End-to-End

```python
import pandas as pd
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor

from salasml.pipeline import num_pipe, cat_pipe
from salasml.utils import save_model, load_model
from salasml.plot import plot_missing_value
from salasml.feature_importance import mean_score_decrease
from salasml.tuning import random_search_params as rsp

# 1. Load data
df = pd.read_csv("data.csv", index_col="id")

# 2. Cek missing value
plot_missing_value(df, return_df=True)

# 3. Split
X = df.drop(columns="target")
y = df["target"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. Preprocessor
num_cols = ['age', 'income']
cat_cols = ['gender', 'city']

preprocessor = ColumnTransformer([
    ('numeric',   num_pipe(scaling='standard'), num_cols),
    ('categoric', cat_pipe(encoder='onehot'),   cat_cols),
])

# 5. Pipeline
pipeline = Pipeline([
    ('prep', preprocessor),
    ('algo', RandomForestRegressor(n_jobs=-1, random_state=42))
])

# 6. Tuning
model = RandomizedSearchCV(pipeline, rsp.rf_params, cv=5, scoring='r2',
                           n_iter=50, n_jobs=-1, verbose=1, random_state=42)
model.fit(X_train, y_train)

print(model.best_params_)
print(model.score(X_train, y_train), model.best_score_, model.score(X_test, y_test))

# 7. Feature importance
mean_score_decrease(X_train, y_train, model, plot=True, topk=10)

# 8. Save
save_model(model.best_estimator_, "rf_model.pkl")
```

---

## VS Code / Kiro Snippets

Tersedia snippet untuk VS Code dan Kiro IDE di folder `snippets/`.

### Cara install snippet di VS Code / Kiro

1. Buka Command Palette → `Preferences: Configure User Snippets`
2. Pilih `New Global Snippets file`
3. Copy isi file `snippets/salas.code-snippets` ke file tersebut
4. Simpan

Atau langsung copy file ke folder `.vscode/` di project kamu.

### Cara pakai

Ketik prefix di cell notebook lalu tekan `Tab` atau `Ctrl+Space`:

| Prefix | Keterangan |
|---|---|
| `salas-import` | Import semua package |
| `salas-csv` | Load dataset CSV |
| `salas-split` | Shuffle Split |
| `salas-split-strat` | Stratified Split |
| `salas-prep` | Preprocessor Common |
| `salas-prep-adv` | Preprocessor Advance |
| `salas-reg-knn` | Regression KNN |
| `salas-reg-svm` | Regression SVM |
| `salas-reg-rf` | Regression Random Forest |
| `salas-reg-xgb` | Regression XGBoost |
| `salas-reg-linear` | Linear Regression |
| `salas-reg-enet` | ElasticNet |
| `salas-clf-knn` | Classification KNN |
| `salas-clf-svm` | Classification SVM |
| `salas-clf-rf` | Classification Random Forest |
| `salas-clf-xgb` | Classification XGBoost |
| `salas-clf-logistic` | Logistic Regression |
| `salas-tune-grid` | Grid Search |
| `salas-tune-random` | Randomized Search |
| `salas-tune-bayes` | Bayesian Search |
| `salas-save` | Save model |
| `salas-save-best` | Save best estimator |

---

## Lisensi

MIT License
