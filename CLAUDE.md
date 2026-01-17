# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

DaskRahaBaran is a Dask-based parallel implementation of the Raha (error detection) and Baran (error correction) data cleaning systems. The key innovation is a shared memory model using `multiprocessing.shared_memory` that allows concurrent workers to access data without duplication, combined with Dask for task parallelism.

## Installation & Setup

```bash
# Requires Python 3.10
pip install -e .[dask]
```

## Running the System

**Primary interface**: Jupyter notebooks in `/daskraha/`
- `pipeline_1_(minimal_and_sequential).ipynb` - Basic end-to-end pipeline
- `pipeline_2_(minimal_and_integrated).ipynb` - Integrated approach
- `pipeline_3_(detailed_demo).ipynb` - Advanced demonstration

**Programmatic usage**:
```python
from daskraha.dask_version.detection_parallel import DetectionParallel
from daskraha.dask_version.correction_parallel import CorrectionParallel

dataset_dict = {
    "name": "dataset_name",
    "path": "/path/to/dirty.csv",
    "clean_path": "/path/to/clean.csv"  # optional, for evaluation
}

# Error Detection (Raha)
detect = DetectionParallel()
detect.LABELING_BUDGET = 20
client = detect.start_dask_cluster(num_workers=4)
detected_cells = detect.run(dataset_dict)

# Error Correction (Baran)
correct = CorrectionParallel()
correct.LABELING_BUDGET = 20
corrected_cells = correct.run(dataset_dict)
```

## Architecture

### Three-Layer Structure
```
User Interface Layer (Jupyter Notebooks / Python API)
         ↓
Application Layer (DetectionParallel / CorrectionParallel)
         ↓
Parallel Execution Layer (Dask LocalCluster + Shared Memory)
```

### Core Modules (`/daskraha/`)

**Abstract base classes**:
- `detection.py` - Detection interface (run_strategies, generate_features, build_clusters, sample_tuple, propagate_labels, predict_labels)
- `correction.py` - Correction interface (initialize_models, generate_features, sample_tuple, update_models, predict_corrections)
- `dataset.py` - Dataset management with evaluation utilities
- `constants.py` - Detection algorithm constants: OD, PVD, RVD, KBVD

**Parallel implementations** (`/daskraha/dask_version/`):
- `detection_parallel.py` (~1100 lines) - Raha with Dask parallelism
- `correction_parallel.py` (~900 lines) - Baran with Dask parallelism
- `dataset_parallel.py` (~500 lines) - SharedDataFrame, SharedNumpyArray, DatasetParallel
- `container.py` - Module-level shared memory references

### Shared Memory Model

Critical for performance. Uses `multiprocessing.shared_memory.SharedMemory` with naming convention:
- `"dirty-xyabc"` - Dirty DataFrame (5-char hash)
- `"clean-xyabc"` - Clean DataFrame
- `"xyabc-f-r-c0"` - Feature vectors for column 0
- `"xyabc-s_p-c0"` - Strategy profiles for column 0

Requires fork multiprocessing:
```python
dask.config.set({"distributed.worker.multiprocessing-method": "fork"})
```

### Detection Strategies (Raha)
1. **OD (Outlier Detection)** - Statistical anomalies via dBoost
2. **PVD (Pattern Violation Detection)** - Character pattern mismatches
3. **RVD (Rule Violation Detection)** - Functional dependency violations
4. **KBVD (Knowledge-Base Violation Detection)** - KATARA integration

### Correction Models (Baran)
1. **Value-based** - Learned transformations (remover, adder, replacer, swapper)
2. **Vicinity-based** - Functional dependencies (j1→j2 mappings)
3. **Domain-based** - Column value frequencies

## Pipeline Flow

```
Dirty Data → Raha (DetectionParallel.run())
    ├─ Run 4 detection strategies in parallel
    ├─ Generate features + cluster per column
    ├─ Interactive sampling + label propagation
    └─ Output: detected_cells {(row, col): ""}
                    ↓
             Baran (CorrectionParallel.run())
    ├─ Initialize correction models
    ├─ Iterative sampling + model updates
    ├─ Generate corrections + predict
    └─ Output: corrected_cells {(row, col): "value"}
```

## Key Parameters

```python
# Detection (Raha)
app.LABELING_BUDGET = 20                    # Tuples to label
app.CLUSTERING_BASED_SAMPLING = True        # Uncertainty-based sampling
app.LABEL_PROPAGATION_METHOD = "homogeneity"  # Or "majority"
app.CLASSIFICATION_MODEL = "GBC"            # Gradient Boosting
app.ERROR_DETECTION_ALGORITHMS = [OD, PVD, RVD, KBVD]

# Correction (Baran)
app.LABELING_BUDGET = 20
app.CLASSIFICATION_MODEL = "ABC"            # AdaBoost
app.MIN_CORRECTION_OCCURRENCE = 2
app.NUM_WORKERS = cpu_count()
```

## External Tools

- **dBoost** (`/daskraha/dask_version/tools/dBoost/`) - Outlier detection subprocess
- **KATARA** (`/daskraha/dask_version/tools/KATARA/`) - Knowledge-base violations
- **Knowledge base** (`/supplementaries/knowledge-base/`) - ~500 relation files

## Sample Datasets

Located in `/datasets/`: flights, hospital, beers, movies_1, rayyan, tax, toy

Each dataset folder contains `dirty.csv` and `clean.csv` files.

## Debugging

- Set `app.VERBOSE = True` for detailed logs
- Check `dataset.results_folder` for intermediate results
- Monitor Dask dashboard at `localhost:8787`
