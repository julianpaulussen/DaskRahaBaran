import sys
import shutil
import tempfile
import multiprocessing as mp
from pathlib import Path
from collections import defaultdict

import numpy as np
import flwr as fl
from flwr.common import ndarrays_to_parameters, parameters_to_ndarrays

project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

import daskraha.dask_version.dataset_parallel as dp


# ---------------------------------------------------------------------------
# Helper: run Raha in a subprocess (same pattern as evaluate_local_raha_baseline)
# ---------------------------------------------------------------------------
def _run_raha_subprocess(project_root_str, data_dict, labeling_budget, result_queue):
    sys.path.insert(0, project_root_str)
    import daskraha.dask_version.detection_parallel as det_p

    detector = det_p.DetectionParallel()
    detector.VERBOSE = False
    detector.CLASSIFICATION_MODEL = "SGDC"
    detector.LABELING_BUDGET = labeling_budget
    try:
        result = detector.run(data_dict)
        result_queue.put(result)
    except Exception as e:
        result_queue.put(e)


# ---------------------------------------------------------------------------
# Encode / decode weights for Flower transport
# ---------------------------------------------------------------------------
# Flower transports a flat list of numpy arrays. We encode per-column weights
# plus metadata (which columns have weights, which have fallbacks) into this
# format.
#
# Layout:
#   [0]    = meta array: for each column, 0=weights, 1=fallback_0, 2=fallback_1
#   [1..N] = pairs of (coef, intercept) for columns that have weights
# ---------------------------------------------------------------------------

def encode_weights(classifier_weights, fallback_labels, num_columns):
    """Encode per-column classifier weights and fallbacks into a list of ndarrays."""
    meta = np.zeros(num_columns, dtype=np.float64)
    weight_arrays = []

    for col_idx in range(num_columns):
        if col_idx in classifier_weights:
            meta[col_idx] = 0  # has real weights
            coef, intercept = classifier_weights[col_idx]
            weight_arrays.append(coef.flatten())
            weight_arrays.append(intercept.flatten())
        elif col_idx in fallback_labels:
            meta[col_idx] = 1 + fallback_labels[col_idx]  # 1=fallback_0, 2=fallback_1
        else:
            meta[col_idx] = -1  # unknown

    return [meta] + weight_arrays


def decode_weights(ndarrays, num_columns):
    """Decode a list of ndarrays back into per-column weights and fallbacks."""
    meta = ndarrays[0]
    classifier_weights = {}
    fallback_labels = {}

    weight_idx = 1
    for col_idx in range(num_columns):
        if meta[col_idx] == 0:  # real weights
            coef = ndarrays[weight_idx].reshape(1, -1)
            intercept = ndarrays[weight_idx + 1]
            classifier_weights[col_idx] = (coef, intercept)
            weight_idx += 2
        elif meta[col_idx] == 1:
            fallback_labels[col_idx] = 0
        elif meta[col_idx] == 2:
            fallback_labels[col_idx] = 1

    return classifier_weights, fallback_labels


# ---------------------------------------------------------------------------
# Flower Client
# ---------------------------------------------------------------------------
class RahaFlowerClient(fl.client.NumPyClient):
    def __init__(self, client_id, data_dict, labeling_budget, num_columns):
        self.client_id = client_id
        self.data_dict = data_dict
        self.labeling_budget = labeling_budget
        self.num_columns = num_columns
        self.classifier_weights = {}
        self.fallback_labels = {}
        self.detected_cells = {}

    def get_parameters(self, config):
        return encode_weights(self.classifier_weights, self.fallback_labels, self.num_columns)

    def fit(self, parameters, config):
        """Run local Raha detection and return classifier weights."""
        # Clean up any leftover tmp dir
        tmp_dir = Path(tempfile.gettempdir()) / self.data_dict["name"]
        if tmp_dir.exists():
            shutil.rmtree(tmp_dir)

        # Run Raha in a subprocess
        ctx = mp.get_context("spawn")
        result_queue = ctx.Queue()
        p = ctx.Process(
            target=_run_raha_subprocess,
            args=(str(project_root), self.data_dict, self.labeling_budget, result_queue),
        )
        p.start()
        p.join()
        result = result_queue.get()

        if isinstance(result, Exception):
            raise result

        self.detected_cells = result["detected_cells"]
        self.classifier_weights = result["classifier_weights"]
        self.fallback_labels = result["fallback_labels"]

        encoded = encode_weights(self.classifier_weights, self.fallback_labels, self.num_columns)
        num_samples = len(self.detected_cells)

        return encoded, num_samples, {}

    def evaluate(self, parameters, config):
        # Not used in our flow — evaluation happens globally
        return 0.0, 0, {}


# ---------------------------------------------------------------------------
# Custom Flower Strategy: FedAvg on weights, majority vote on fallbacks
# ---------------------------------------------------------------------------
class RahaFedAvg(fl.server.strategy.FedAvg):
    def __init__(self, num_columns, **kwargs):
        super().__init__(**kwargs)
        self.num_columns = num_columns
        self.global_weights = {}
        self.global_fallbacks = {}

    def aggregate_fit(self, server_round, results, failures):
        if not results:
            return None, {}

        # Collect per-column weights and fallbacks from all clients
        all_weights = defaultdict(list)    # col_idx -> list of (coef, intercept)
        all_fallbacks = defaultdict(list)  # col_idx -> list of 0/1

        for _, fit_res in results:
            ndarrays = parameters_to_ndarrays(fit_res.parameters)
            client_weights, client_fallbacks = decode_weights(ndarrays, self.num_columns)

            for col_idx, (coef, intercept) in client_weights.items():
                all_weights[col_idx].append((coef, intercept))
            for col_idx, label in client_fallbacks.items():
                all_fallbacks[col_idx].append(label)

        # Aggregate per column
        aggregated_weights = {}
        aggregated_fallbacks = {}

        for col_idx in range(self.num_columns):
            if col_idx in all_weights and len(all_weights[col_idx]) > 0:
                # FedAvg: average coef and intercept
                coefs = [w[0] for w in all_weights[col_idx]]
                intercepts = [w[1] for w in all_weights[col_idx]]
                avg_coef = np.mean(coefs, axis=0)
                avg_intercept = np.mean(intercepts, axis=0)
                aggregated_weights[col_idx] = (avg_coef, avg_intercept)
            elif col_idx in all_fallbacks and len(all_fallbacks[col_idx]) > 0:
                # Majority vote on fallbacks
                aggregated_fallbacks[col_idx] = int(np.round(np.mean(all_fallbacks[col_idx])))
            # else: no info from any client for this column

        self.global_weights = aggregated_weights
        self.global_fallbacks = aggregated_fallbacks

        # Encode global result for distribution back to clients
        encoded = encode_weights(aggregated_weights, aggregated_fallbacks, self.num_columns)
        parameters = ndarrays_to_parameters(encoded)

        return parameters, {}


# ---------------------------------------------------------------------------
# Run federated experiment
# ---------------------------------------------------------------------------
def run_federated(dataset_name, n_clients, per_node_budget, num_columns):
    """Run a federated Raha experiment using Flower simulation."""
    base_path = project_root / "datasets" / dataset_name
    split_dir = base_path / f"split_{n_clients}"

    if not split_dir.exists():
        print(f"Error: Split directory {split_dir} does not exist.")
        return

    def client_fn(cid):
        client_id = int(cid) + 1
        data_dict = {
            "name": f"{dataset_name}_split_{client_id}",
            "path": str((split_dir / f"dirty_{client_id}.csv").resolve()),
            "clean_path": str((split_dir / f"clean_{client_id}.csv").resolve()),
        }
        return RahaFlowerClient(client_id, data_dict, per_node_budget, num_columns).to_client()

    strategy = RahaFedAvg(
        num_columns=num_columns,
        min_fit_clients=n_clients,
        min_available_clients=n_clients,
    )

    fl.simulation.start_simulation(
        client_fn=client_fn,
        num_clients=n_clients,
        config=fl.server.ServerConfig(num_rounds=1),
        strategy=strategy,
    )

    print("\n" + "=" * 50)
    print(f"FEDERATED AGGREGATION COMPLETE ({n_clients} clients)")
    print("=" * 50)
    print(f"Columns with averaged weights: {list(strategy.global_weights.keys())}")
    print(f"Columns with fallback labels:  {list(strategy.global_fallbacks.keys())}")

    return strategy.global_weights, strategy.global_fallbacks


if __name__ == "__main__":
    dataset = "hospital"
    n_clients = 2
    budget = 20
    num_columns = 20  # hospital has 20 columns

    run_federated(dataset, n_clients, budget, num_columns)
