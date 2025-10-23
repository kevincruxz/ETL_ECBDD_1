import argparse
import pandas as pd
import sys
from pathlib import Path

NUMERIC_COLS_DEFAULT = [
    "metacritic_score",
    "metacritic_rating_count",
    "metacritic_user_score",
    "metacritic_user_rating_count",
    "playstation_score",
    "playstation_rating_count",
]

def parse_args():
    p = argparse.ArgumentParser(description="Limpia un dataset de videojuegos y genera un CSV limpio.")
    p.add_argument("input_csv", help="Ruta del archivo CSV de entrada")
    p.add_argument("-o", "--output_csv", help="Ruta del CSV de salida (por defecto: junto al input con sufijo _limpio)")
    p.add_argument("--price-col", default="highest_price", help='Nombre de la columna de precio (default: "highest_price")')
    p.add_argument("--date-col", default="release_date", help='Nombre de la columna de fecha (default: "release_date")')
    p.add_argument("--genre-col", default="genre", help='Nombre de la columna de género (default: "genre")')
    p.add_argument("--name-col", default="game_name", help='Nombre de la columna de nombre (default: "game_name")')
    p.add_argument("--platform-col", default="platform", help='Nombre de la columna de plataforma (default: "platform")')
    p.add_argument("--numeric-cols", nargs="*", default=NUMERIC_COLS_DEFAULT,
                   help="Columnas que deben ser numéricas (se convierte con coerción).")
    return p.parse_args()

def clean_price_series(s: pd.Series) -> pd.Series:
    s = s.astype(str).str.replace(r"[€$]", "", regex=True)
    s = s.str.replace(",", "", regex=False).str.strip()
    return pd.to_numeric(s, errors="coerce")

def main():
    args = parse_args()
    input_path = Path(args.input_csv)
    if not input_path.exists():
        print(f"ERROR: no se encontró el archivo: {input_path}", file=sys.stderr)
        sys.exit(1)

    output_path = (
        Path(args.output_csv)
        if args.output_csv
        else input_path.with_name(f"{input_path.stem}_limpio{input_path.suffix}")
    )

    df = pd.read_csv(input_path)

    df = df.drop_duplicates()

    if args.genre_col in df.columns:
        df[args.genre_col] = df[args.genre_col].replace("--", pd.NA)

    if args.price_col in df.columns:
        df[args.price_col] = clean_price_series(df[args.price_col])

    if args.date_col in df.columns:
        df[args.date_col] = pd.to_datetime(df[args.date_col], errors="coerce")

    for col in args.numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    cols_requeridas = [c for c in [args.name_col, args.platform_col] if c in df.columns]
    if cols_requeridas:
        df = df.dropna(subset=cols_requeridas)

    if args.date_col in df.columns:
        df = df.sort_values(by=args.date_col, ascending=True, na_position="last")

    df.to_csv(output_path, index=False)

    print("=== REPORTE DE LIMPIEZA ===")
    print(f"Archivo de entrada : {input_path}")
    print(f"Archivo de salida  : {output_path}")
    print(f"Filas x Columnas   : {df.shape[0]} x {df.shape[1]}")
    print("\nNulos por columna (top 10):")
    nulls = df.isna().sum().sort_values(ascending=False)
    print(nulls.head(10).to_string())
    # Preview
    print("\nVista previa (5 filas):")
    with pd.option_context("display.max_columns", None, "display.width", 120):
        print(df.head(5))

if __name__ == "__main__":
    main()