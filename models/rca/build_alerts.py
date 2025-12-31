import argparse
import os
from typing import List

import numpy as np
import pandas as pd


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def load_features(path: str) -> pd.DataFrame:
    df = pd.read_parquet(path) if path.endswith(".parquet") else pd.read_csv(path)

    required = {"run_id", "service", "window_start", "window_end"}
    missing = sorted(list(required - set(df.columns)))
    if missing:
        raise ValueError(
            f"Features debe contener {sorted(required)}. Faltan: {missing}"
        )

    df["window_start"] = pd.to_datetime(df["window_start"], utc=True, errors="coerce")
    df["window_end"] = pd.to_datetime(df["window_end"], utc=True, errors="coerce")
    if df["window_start"].isna().any() or df["window_end"].isna().any():
        raise ValueError("No se pudieron parsear window_start/window_end.")

    return df


def select_tfidf_cols(df: pd.DataFrame) -> List[str]:
    # columnas tipo tfidf_0..tfidf_49
    cols = [
        c
        for c in df.columns
        if c.startswith("tfidf_") and pd.api.types.is_numeric_dtype(df[c])
    ]
    return sorted(
        cols, key=lambda x: int(x.split("_")[1]) if x.split("_")[1].isdigit() else x
    )


def build_alerts(
    df: pd.DataFrame,
    *,
    use_metric_rule: bool,
    metric_col: str,
    metric_thr: float,
    use_log_rule: bool,
    log_err_col: str,
    log_kw_col: str,
    log_err_thr: float,
    log_kw_thr: float,
    use_pred_rule: bool,
    pred_prob_col: str,
    pred_thr: float,
) -> pd.DataFrame:
    out_rows = []

    # normaliza NaNs
    df = df.copy()
    for c in [metric_col, log_err_col, log_kw_col, pred_prob_col]:
        if c in df.columns and pd.api.types.is_numeric_dtype(df[c]):
            df[c] = df[c].replace([np.inf, -np.inf], np.nan).fillna(0.0)

    # timestamp “evento”
    ts = df["window_end"]

    # Regla 1: Métrica (ej: error_ratio_mean_1min >= 0.10)
    if use_metric_rule:
        if metric_col not in df.columns:
            raise ValueError(f"No existe metric_col={metric_col} en features.")
        score = df[metric_col].astype(float)
        fired = score >= metric_thr
        if fired.any():
            a = df.loc[
                fired, ["run_id", "service", "window_start", "window_end"]
            ].copy()
            a["ts"] = ts.loc[fired]
            a["signal"] = "metric_rule"
            a["score"] = score.loc[fired].astype(float)
            a["details"] = [f"{metric_col}>={metric_thr}"] * len(a)
            out_rows.append(a)

    # Regla 2: Logs (ej: logs_error_count_1min > 0, keywords > 0)
    if use_log_rule:
        for col in [log_err_col, log_kw_col]:
            if col not in df.columns:
                raise ValueError(f"No existe log col={col} en features.")
        err = df[log_err_col].astype(float)
        kw = df[log_kw_col].astype(float)
        fired = (err >= log_err_thr) | (kw >= log_kw_thr)
        if fired.any():
            # score simple: combina ambas señales
            score = err + kw
            a = df.loc[
                fired, ["run_id", "service", "window_start", "window_end"]
            ].copy()
            a["ts"] = ts.loc[fired]
            a["signal"] = "log_rule"
            a["score"] = score.loc[fired].astype(float)
            a["details"] = [
                f"{log_err_col}>={log_err_thr} OR {log_kw_col}>={log_kw_thr}"
            ] * len(a)
            out_rows.append(a)

    # Regla 3: Predicción
    if use_pred_rule:
        if pred_prob_col not in df.columns:
            raise ValueError(f"No existe pred_prob_col={pred_prob_col} en features.")
        prob = df[pred_prob_col].astype(float)
        fired = prob >= pred_thr
        if fired.any():
            a = df.loc[
                fired, ["run_id", "service", "window_start", "window_end"]
            ].copy()
            a["ts"] = ts.loc[fired]
            a["signal"] = "prediction"
            a["score"] = prob.loc[fired].astype(float)
            a["details"] = [f"{pred_prob_col}>={pred_thr}"] * len(a)
            out_rows.append(a)

    if not out_rows:
        return pd.DataFrame(
            columns=[
                "run_id",
                "service",
                "window_start",
                "window_end",
                "ts",
                "signal",
                "score",
                "details",
            ]
        )

    alerts = pd.concat(out_rows, ignore_index=True)

    # compacta duplicados
    alerts = alerts.groupby(["run_id", "service", "ts", "signal"], as_index=False).agg(
        window_start=("window_start", "min"),
        window_end=("window_end", "max"),
        score=("score", "max"),
        details=("details", "first"),
    )

    # ordena
    alerts = alerts.sort_values(["run_id", "ts", "service", "signal"]).reset_index(
        drop=True
    )
    return alerts


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--features", required=True, help="features_*.parquet/csv")
    p.add_argument("--out-dir", default="data/processed", help="output dir")
    p.add_argument(
        "--run-id", default=None, help="filtra un run_id específico (opcional)"
    )

    # metric rule
    p.add_argument("--use-metric-rule", action="store_true")
    p.add_argument("--metric-col", default="error_ratio_mean_1min")
    p.add_argument("--metric-thr", type=float, default=0.10)

    # log rule
    p.add_argument("--use-log-rule", action="store_true")
    p.add_argument("--log-err-col", default="logs_error_count_1min")
    p.add_argument("--log-kw-col", default="logs_error_keywords_1min")
    p.add_argument("--log-err-thr", type=float, default=1.0)
    p.add_argument("--log-kw-thr", type=float, default=1.0)

    # prediction rule (opcional)
    p.add_argument("--use-pred-rule", action="store_true")
    p.add_argument("--pred-prob-col", default="pred_prob")
    p.add_argument("--pred-threshold", type=float, default=0.45)

    args = p.parse_args()
    ensure_dir(args.out_dir)

    df = load_features(args.features)
    if args.run_id:
        df = df[df["run_id"].astype(str) == str(args.run_id)].copy()

    alerts = build_alerts(
        df,
        use_metric_rule=args.use_metric_rule,
        metric_col=args.metric_col,
        metric_thr=args.metric_thr,
        use_log_rule=args.use_log_rule,
        log_err_col=args.log_err_col,
        log_kw_col=args.log_kw_col,
        log_err_thr=args.log_err_thr,
        log_kw_thr=args.log_kw_thr,
        use_pred_rule=args.use_pred_rule,
        pred_prob_col=args.pred_prob_col,
        pred_thr=args.pred_threshold,
    )

    run_id = (
        str(args.run_id)
        if args.run_id
        else (str(df["run_id"].iloc[0]) if len(df) else "unknown_run")
    )

    out_path = os.path.join(args.out_dir, f"alerts_{run_id}.parquet")
    alerts.to_parquet(out_path, index=False)
    print(f"[ok] saved alerts: {out_path} (rows={len(alerts)})")
    print(alerts.head(10).to_string(index=False))


if __name__ == "__main__":
    main()
