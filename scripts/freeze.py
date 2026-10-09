"""Freeze the dev-split choices. Reads the tuning tables, writes configs/frozen.yaml.

Rule (stated before the test split was touched): for each detector, the fixed setting is the
one with the highest mean pooled F1 across all ten resolution levels (IoU 0.25, area
method) on the dev split. The per-level setting is the best dev setting at each level.

    .venv/bin/python scripts/freeze.py
"""
import pandas as pd
import yaml

TABLES = {"owlvit": "results/tune_owlvit.csv", "owlv2": "results/tune_owlv2.csv",
          "colour": "results/tune_colour.csv"}


def main() -> None:
    frozen = {"rule": "highest mean dev F1 over levels, IoU 0.25, area method", "detectors": {}}
    for det, path in TABLES.items():
        df = pd.read_csv(path)
        mean_f1 = df.groupby("config")["f1"].mean().sort_values(ascending=False)
        per_level = df.loc[df.groupby("level")["f1"].idxmax()].set_index("level")["config"]
        frozen["detectors"][det] = {
            "fixed": mean_f1.index[0],
            "fixed_dev_mean_f1": round(float(mean_f1.iloc[0]), 3),
            "runner_up": [[k, round(float(v), 3)] for k, v in mean_f1.iloc[1:5].items()],
            "per_level": {float(k): v for k, v in per_level.items()},
        }
        print(det, "->", mean_f1.index[0], round(float(mean_f1.iloc[0]), 2))
    with open("configs/frozen.yaml", "w") as fh:
        yaml.safe_dump(frozen, fh, sort_keys=False)
    print("wrote configs/frozen.yaml")


if __name__ == "__main__":
    main()
