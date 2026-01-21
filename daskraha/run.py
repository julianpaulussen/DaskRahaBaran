#!/usr/bin/env python3
from pathlib import Path
import os, logging
import requests
import daskraha.dask_version.detection_parallel as dp
from daskraha.dask_version import container


def label_interactive(app, dataset_par):
    df = container.shared_dataframe.read()
    while len(dataset_par.labeled_tuples) < app.LABELING_BUDGET:
        app.sample_tuple(dataset_par)
        row = dataset_par.sampled_tuple
        print(f"\n{'='*60}")
        if row > 0:
            print(f"--- Row {row-1} (before) ---")
            for j, col in enumerate(df.columns):
                print(f"  {col}: {df.iloc[row-1, j]}")
        print(f"\n>>> Row {row} (current) <<<")
        for j, col in enumerate(df.columns):
            print(f"  {col}: {df.iloc[row, j]}")
        if row < len(df) - 1:
            print(f"\n--- Row {row+1} (after) ---")
            for j, col in enumerate(df.columns):
                print(f"  {col}: {df.iloc[row+1, j]}")
        print()
        for j, col in enumerate(df.columns):
            val = df.iloc[row, j]
            corr = input(f"{col} ('{val}'): ").strip()
            dataset_par.labeled_cells[(row, j)] = [1 if corr else 0, corr or val]
        dataset_par.labeled_tuples[row] = 1


def evaluate_with_dias(csv_content):
    url = "https://dias-479914.appspot.com/DataCleaningEvaluator/"
    r = requests.post(url, data={"value": csv_content})
    if r.status_code != 200:
        print(f"Error: {r.status_code}")
    print(r.text)


def main():
    base_path = Path(__file__).parent.parent / "datasets" / "dias"
    dataset_dict = {
        "name": "dias",
        "path": str(base_path / "dirty.csv"),
        "clean_path": str(base_path / "dirty.csv"),  # dummy for init
    }

    app = dp.DetectionParallel()
    app.LABELING_BUDGET = 10
    app.VERBOSE = True

    # Setup
    app.initialize_dataframe(dataset_dict["path"])
    client = app.start_dask_cluster(num_workers=os.cpu_count(), logging_level=logging.ERROR)
    client.run(app.init_workers)
    dataset_par, _ = app.initialize_dataset(dataset_dict)

    # Raha pipeline
    app.run_strategies(dataset_par)
    app.generate_features(dataset_par)
    app.build_clusters(dataset_par)
    label_interactive(app, dataset_par)
    app.propagate_labels(dataset_par)
    app.predict_labels(dataset_par)

    # Results
    print(f"\nDetected {len(dataset_par.detected_cells)} error cells")

    # Create output CSV with "xxx" for detected errors
    df = container.shared_dataframe.read().copy()
    for (row, col) in dataset_par.detected_cells:
        df.iloc[row, col] = "xxx"

    # Save to clean.csv
    output_path = base_path / "clean.csv"
    df.to_csv(output_path, index=False)
    print(f"\nSaved to {output_path}")

    # Evaluate via DIAS web service
    csv_content = df.to_csv(index=False)
    print("Sending to DIAS evaluator...")
    evaluate_with_dias(csv_content)

    app.cleanup_raha(dataset_par)
    client.shutdown()


if __name__ == "__main__":
    main()