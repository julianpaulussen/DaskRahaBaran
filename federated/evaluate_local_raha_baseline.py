import os
import sys
import shutil
import tempfile
import multiprocessing as mp
from pathlib import Path
import numpy as np
import pandas as pd

# Ensure project root is in path for imports
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

import daskraha.dask_version.dataset_parallel as dp
import daskraha.dask_version.detection_parallel as det_p


def _run_client_subprocess(project_root_str, data_dict, labeling_budget, result_queue):
    """
    Runs detector.run() in a fresh spawned process to avoid the fork-after-threading
    deadlock that occurs when a new Dask LocalCluster is created after a prior one
    was shut down in the same process.
    """
    sys.path.insert(0, project_root_str)
    import daskraha.dask_version.detection_parallel as det_p

    detector = det_p.DetectionParallel()
    detector.VERBOSE = False
    detector.LABELING_BUDGET = labeling_budget
    try:
        detected_cells = detector.run(data_dict)
        result_queue.put(detected_cells)
    except Exception as e:
        result_queue.put(e)


def run_local_raha_baseline(dataset_name, n_clients=2, total_label_budget=None, per_node_budget=None):
    """
    Runs Local-only Raha on subsets and evaluates concatenated results.
    Uses saved .npy index files to map local results back to the global dataset.
    Each client runs in an isolated spawned subprocess to avoid Dask fork deadlocks.

    per_node_budget: fixed labeling budget per node (takes precedence over total_label_budget)
    total_label_budget: divided evenly across all clients (legacy parameter)
    """
    base_path = project_root / "datasets" / dataset_name
    split_dir = base_path / f"split_{n_clients}"

    if not split_dir.exists():
        print(f"Error: Split directory {split_dir} does not exist. Run split_dataset.py first.")
        return

    if per_node_budget is not None:
        per_client_budget = per_node_budget
        print(f"Label budget: {per_client_budget} per node")
    elif total_label_budget is not None:
        per_client_budget = max(1, total_label_budget // n_clients)
        print(f"Label budget: {total_label_budget} total → {per_client_budget} per client")
    else:
        per_client_budget = 20

    # This will hold ALL detected errors from ALL nodes
    global_detected_errors = {}

    print(f"--- Running Local Raha Baseline on {dataset_name} ({n_clients} clients) ---")

    ctx = mp.get_context("spawn")

    for client_id in range(1, n_clients + 1):
        print(f"\n>>> Processing Client {client_id}...")

        dirty_path = split_dir / f"dirty_{client_id}.csv"
        indices_path = split_dir / f"indices_{client_id}.npy"

        if not dirty_path.exists() or not indices_path.exists():
            print(f"Warning: Missing files for client {client_id}. Skipping...")
            continue

        # Load global index mapping
        global_indices = np.load(indices_path)

        data_dict = {
            "name": f"{dataset_name}_split_{client_id}",
            "path": str(dirty_path.resolve()),
            "clean_path": str((split_dir / f"clean_{client_id}.csv").resolve())
        }

        # Clean up any leftover tmp dir from a previous run
        tmp_dir = Path(tempfile.gettempdir()) / f"{dataset_name}_split_{client_id}"
        if tmp_dir.exists():
            shutil.rmtree(tmp_dir)

        try:
            result_queue = ctx.Queue()
            p = ctx.Process(
                target=_run_client_subprocess,
                args=(str(project_root), data_dict, per_client_budget, result_queue),
            )
            p.start()
            p.join()
            result = result_queue.get()

            if isinstance(result, Exception):
                raise result

            detected_cells = result

            # Map results back to global (row, col)
            for (local_row, col), val in detected_cells.items():
                global_row = global_indices[local_row]
                global_detected_errors[(global_row, col)] = val

            print(f"Client {client_id} complete.")
        except Exception as e:
            print(f"Error processing Client {client_id}: {e}")

    print("\n" + "="*50)
    print("FINAL EVALUATION (Concatenated 1,000-row Score)")
    print("="*50)

    # Evaluate using the full dataset object
    full_dataset = dp.DatasetParallel({
        "name": dataset_name,
        "path": str((base_path / "dirty.csv").resolve()),
        "clean_path": str((base_path / "clean.csv").resolve())
    })

    metrics = full_dataset.get_data_cleaning_evaluation(global_detected_errors)
    p, r, f1 = metrics[:3]

    print(f"Concatenated Precision: {p:.4f}")
    print(f"Concatenated Recall:    {r:.4f}")
    print(f"Concatenated F1 Score:  {f1:.4f}")

if __name__ == "__main__":
    dataset = "hospital"
    clients = 10
    # Set total budget to None if each node should get the full budget (20)
    total_budget = 20 
    run_local_raha_baseline(dataset, clients, total_budget)

