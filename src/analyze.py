from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


BASE_DIR = Path(__file__).resolve().parents[1]
DATA_PATH = BASE_DIR / "data" / "sample" / "sensor_readings.csv"
OUTPUT_DIR = BASE_DIR / "output"

NUMERIC_COLUMNS = [
    "temperature_c",
    "vibration_mm_s",
    "pressure_bar",
    "rpm",
    "load_pct",
]


def load_data() -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["machine_id"] = df["machine_id"].astype(str).str.strip().str.upper()
    df = df.sort_values(["machine_id", "timestamp"]).reset_index(drop=True)
    return df


def calculate_anomalies(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    for column in NUMERIC_COLUMNS:
        group_mean = df.groupby("machine_id")[column].transform("mean")
        group_std = df.groupby("machine_id")[column].transform("std").replace(0, 1)
        df[f"{column}_zscore"] = ((df[column] - group_mean) / group_std).abs()

    zscore_columns = [f"{column}_zscore" for column in NUMERIC_COLUMNS]
    df["anomaly_score"] = df[zscore_columns].sum(axis=1)
    df["is_anomaly"] = (
        (df["temperature_c_zscore"] > 2)
        | (df["vibration_mm_s_zscore"] > 2)
        | (df["pressure_bar_zscore"] > 2)
        | (df["anomaly_score"] > 8)
    )

    return df


def build_health_summary(df: pd.DataFrame) -> pd.DataFrame:
    summary = (
        df.groupby("machine_id")
        .agg(
            reading_count=("machine_id", "count"),
            anomaly_count=("is_anomaly", "sum"),
            avg_temperature_c=("temperature_c", "mean"),
            avg_vibration_mm_s=("vibration_mm_s", "mean"),
            avg_pressure_bar=("pressure_bar", "mean"),
            avg_rpm=("rpm", "mean"),
            avg_load_pct=("load_pct", "mean"),
            max_anomaly_score=("anomaly_score", "max"),
        )
        .reset_index()
    )
    summary["anomaly_rate_pct"] = (summary["anomaly_count"] / summary["reading_count"] * 100).round(2)
    summary = summary.sort_values(["anomaly_count", "max_anomaly_score"], ascending=[False, False])
    return summary


def save_outputs(df: pd.DataFrame, summary: pd.DataFrame) -> None:
    OUTPUT_DIR.mkdir(exist_ok=True)
    anomalies = df[df["is_anomaly"]].copy()
    anomalies.to_csv(OUTPUT_DIR / "flagged_anomalies.csv", index=False)
    summary.to_csv(OUTPUT_DIR / "machine_health_summary.csv", index=False)


def create_plot(df: pd.DataFrame, machine_id: str = "M01") -> None:
    machine_df = df[df["machine_id"] == machine_id].copy()

    fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)

    axes[0].plot(machine_df["timestamp"], machine_df["temperature_c"], marker="o", label="Temperature (C)")
    axes[0].set_title(f"Machine {machine_id} Temperature")
    axes[0].set_ylabel("Temperature (C)")
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(machine_df["timestamp"], machine_df["vibration_mm_s"], marker="o", color="darkred", label="Vibration")
    anomaly_points = machine_df[machine_df["is_anomaly"]]
    axes[1].scatter(
        anomaly_points["timestamp"],
        anomaly_points["vibration_mm_s"],
        color="orange",
        s=70,
        label="Flagged anomaly",
        zorder=3,
    )
    axes[1].set_title(f"Machine {machine_id} Vibration")
    axes[1].set_ylabel("mm/s")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()

    plt.xticks(rotation=30)
    plt.tight_layout()
    OUTPUT_DIR.mkdir(exist_ok=True)
    plt.savefig(OUTPUT_DIR / "machine_m01_signals.png", dpi=150)
    plt.close(fig)


def print_summary(summary: pd.DataFrame, anomalies: pd.DataFrame) -> None:
    print("Predictive maintenance analysis completed successfully.")
    print(f"Anomaly records: {len(anomalies)}")
    print()
    print("Machine health summary:")
    print(summary.to_string(index=False))


def main() -> None:
    df = load_data()
    df = clean_data(df)
    df = calculate_anomalies(df)
    summary = build_health_summary(df)
    save_outputs(df, summary)
    create_plot(df)
    anomalies = df[df["is_anomaly"]].copy()
    print_summary(summary, anomalies)


if __name__ == "__main__":
    main()
