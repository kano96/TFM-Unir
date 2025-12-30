import argparse
import json
import os
from datetime import timedelta

import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def load_features(path: str) -> pd.DataFrame:
    if path.endswith(".parquet"):
        df = pd.read_parquet(path)
    else:
        df = pd.read_csv(path)

    required = {"run_id", "service", "window_start", "window_end"}
    if not required.issubset(df.columns):
        raise ValueError(f"Features deben contener {required}")

    df["window_start"] = pd.to_datetime(df["window_start"], utc=True)
    df["window_end"] = pd.to_datetime(df["window_end"], utc=True)
    return df


def load_labels(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["start_dt"] = pd.to_datetime(df["start_ts"], utc=True)
    df["end_dt"] = pd.to_datetime(df["end_ts"], utc=True)
    return df


def build_future_label(features: pd.DataFrame, labels: pd.DataFrame, horizon_min: int):
    y = []
    horizon = timedelta(minutes=horizon_min)

    for _, r in features.iterrows():
        we = r["window_end"]
        run_id = str(r["run_id"])
        service = str(r["service"])

        future_faults = labels[
            (labels["run_id"].astype(str) == run_id)
            & (labels["service"].astype(str) == service)
            & (labels["start_dt"] >= we)
            & (labels["start_dt"] <= we + horizon)
        ]

        y.append(1 if not future_faults.empty else 0)

    features["y_future"] = y
    return features


def select_feature_columns(df: pd.DataFrame):
    exclude = {"run_id", "service", "window_start", "window_end", "y_future"}
    return [
        c
        for c in df.columns
        if c not in exclude and pd.api.types.is_numeric_dtype(df[c])
    ]


def temporal_split(df: pd.DataFrame, train_ratio=0.7):
    df = df.sort_values("window_start")
    n = int(len(df) * train_ratio)
    return df.iloc[:n], df.iloc[n:]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", required=True)
    parser.add_argument("--labels", required=True)
    parser.add_argument("--horizon-min", type=int, default=5)
    parser.add_argument("--out-dir", default="models/artifacts")
    args = parser.parse_args()

    ensure_dir(args.out_dir)

    feats = load_features(args.features)
    labels = load_labels(args.labels)

    feats = build_future_label(feats, labels, args.horizon_min)
    feature_cols = select_feature_columns(feats)

    train_df, test_df = temporal_split(feats)

    X_train = train_df[feature_cols]
    y_train = train_df["y_future"]
    X_test = test_df[feature_cols]
    y_test = test_df["y_future"]

    model = Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "clf",
                LogisticRegression(
                    max_iter=1000, class_weight="balanced", random_state=42
                ),
            ),
        ]
    )

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    print(classification_report(y_test, y_pred, zero_division=0))

    joblib.dump(model, os.path.join(args.out_dir, "predictor.joblib"))
    with open(os.path.join(args.out_dir, "predictor_features.json"), "w") as f:
        json.dump(feature_cols, f, indent=2)

    print("[ok] Predictor entrenado y guardado")


if __name__ == "__main__":
    main()
