import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor

from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

# 1) Carga de datos
df = pd.read_csv("/mnt/data/dataset_original.csv")

# 2) Limpieza / preparación mínima
#   2.1) Precio: quitar símbolo € y convertir a float
def parse_price(x):
    if pd.isna(x):
        return np.nan
    x = str(x).replace("€", "").replace(",", "").strip()
    try:
        return float(x)
    except:
        return np.nan

df["highest_price_num"] = df["highest_price"].apply(parse_price)

#   2.2) Año de lanzamiento a partir de release_date (ej. "Feb 15, 2012")
def parse_year(s):
    try:
        return pd.to_datetime(s).year
    except:
        return np.nan

df["release_year"] = df["release_date"].apply(parse_year)

# 3) Selección de variable objetivo y variables de entrada
target = "playstation_score"

feature_cols_num = [
    "highest_price_num",
    "release_year",
    "metacritic_score",
    "metacritic_rating_count",
    "metacritic_user_score",
    "metacritic_user_rating_count",
    "playstation_rating_count",
]

feature_cols_cat = [
    "genre",
    "platform",
    "publisher",
]

# Filtra columnas que existan realmente (por si alguna no está)
feature_cols_num = [c for c in feature_cols_num if c in df.columns]
feature_cols_cat = [c for c in feature_cols_cat if c in df.columns]

# Elimina filas sin target
df = df[~df[target].isna()].copy()

X = df[feature_cols_num + feature_cols_cat]
y = df[target].astype(float)

# 4) División Train/Test (80/20)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42
)

# 5) Preprocesamiento: imputación + one-hot para categóricas
numeric_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median"))
])

categorical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
])

preprocess = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, feature_cols_num),
        ("cat", categorical_transformer, feature_cols_cat),
    ],
    remainder="drop"
)

# 6) Modelos
linreg_pipeline = Pipeline(steps=[
    ("prep", preprocess),
    ("model", LinearRegression())
])

rf_pipeline = Pipeline(steps=[
    ("prep", preprocess),
    ("model", RandomForestRegressor(
        n_estimators=300,
        max_depth=None,
        random_state=42,
        n_jobs=-1
    ))
])

# 7) Entrenamiento
linreg_pipeline.fit(X_train, y_train)
rf_pipeline.fit(X_train, y_train)

# 8) Evaluación
def evaluate(name, pipe):
    preds = pipe.predict(X_test)
    r2 = r2_score(y_test, preds)
    mae = mean_absolute_error(y_test, preds)
    rmse = mean_squared_error(y_test, preds, squared=False)
    return pd.Series({"Modelo": name, "R2": r2, "MAE": mae, "RMSE": rmse})

results = pd.DataFrame([
    evaluate("Regresión Lineal", linreg_pipeline),
    evaluate("Random Forest", rf_pipeline),
]).set_index("Modelo")

print(results)

# 9) Importancias de características (solo para RF)
# Recuperamos nombres de columnas después del OneHotEncoder
prep = rf_pipeline.named_steps["prep"]
rf = rf_pipeline.named_steps["model"]

num_names = feature_cols_num
# nombres de categorías tras one-hot
cat_ohe = prep.named_transformers_["cat"].named_steps["onehot"]
cat_names = cat_ohe.get_feature_names_out(feature_cols_cat).tolist()

feature_names = num_names + cat_names
importances = pd.Series(rf.feature_importances_, index=feature_names)
top10 = importances.sort_values(ascending=False).head(10)

print(top10)
