import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

from federated.split_dataset import split_and_save_dataset
from federated.evaluate_local_raha_baseline import run_local_raha_baseline

DATASET = "hospital"
# NODE_COUNTS = [1, 2, 5, 10, 20, 50]
NODE_COUNTS = [1, 2, 5, 10, 20, 50]
# LABELING_BUDGETS = [1, 2, 5, 10, 15, 20]
LABELING_BUDGETS = [5, 10, 20]

if __name__ == "__main__":
    for n in NODE_COUNTS:
        print(f"\n{'='*60}")
        print(f"  Splitting dataset: n={n} nodes")
        print(f"{'='*60}")

        output_dir = project_root / "datasets" / DATASET / f"split_{n}"
        split_and_save_dataset(DATASET, output_dir, n_splits=n)

        for budget in LABELING_BUDGETS:
            print(f"\n{'='*60}")
            print(f"  EXPERIMENT: n={n} nodes, budget={budget} per node")
            print(f"{'='*60}")

            run_local_raha_baseline(DATASET, n_clients=n, per_node_budget=budget)
