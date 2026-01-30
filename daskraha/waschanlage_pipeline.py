"""
Waschanlage Integration for DaskRahaBaran

This module provides an interactive data cleaning pipeline using the waschanlage
UI system for user interaction during the detection and correction phases.
"""

import logging
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from typing_extensions import TypedDict

import numpy
import pandas


# =============================================================================
# TYPE DEFINITIONS FOR UI RETURN VALUES
# =============================================================================

class ConfigurationResult(TypedDict):
    """Return type from configuration UI component."""
    labeling_budget: int


class PipelineConfigurationResult(TypedDict):
    """Return type from full pipeline configuration UI component."""
    datasetName: str
    labelingBudget: int
    pipelineName: str


class LabelingResult(TypedDict):
    """Return type from simple-labeling UI component."""
    labels: List[Dict[str, Any]]  # [{row_id: int, user_label: str}, ...]


class CellLabelingResult(TypedDict):
    """Return type from csv_cell_labeler UI component."""
    is_correct: bool
    correction: Optional[str]

from ui_system import ui

import daskraha.dask_version.dataset_parallel as dp
from daskraha.dask_version import container
from daskraha.dask_version.detection_parallel import DetectionParallel
from daskraha.dask_version.correction_parallel import CorrectionParallel

# Configure project for waschanlage
ui.set_project(
    "DaskRahaBaran",
    tags=["data-cleaning", "raha", "baran", "error-detection", "error-correction"],
    version="1.0.0",
    description="Interactive data cleaning pipeline with Raha (error detection) and Baran (error correction)"
)


# =============================================================================
# CONFIGURATION PHASE
# =============================================================================

@ui.component('pipeline-configuration')
def configure_pipeline(
    existing_datasets: Optional[List[str]] = None,
    default_budget: int = 10,
    default_name: str = "data-cleaning-pipeline"
) -> PipelineConfigurationResult:
    """
    UI Component: Configure the data cleaning pipeline.

    NOTE: This function body never executes. The decorator sends parameters
    to the UI, and the UI's response becomes the return value.

    This component allows the user to:
    - Select or specify a dataset
    - Set the labeling budget for both detection and correction
    - Name the pipeline run

    Args:
        existing_datasets: List of available dataset names (sent to UI)
        default_budget: Default labeling budget 1-100 (sent to UI)
        default_name: Default pipeline name (sent to UI)

    Returns:
        Configuration dict with datasetName, labelingBudget, and pipelineName
        (returned from UI, not from this function body)
    """
    pass  # UI handles this - function body never executes


@ui.component('simple-configurations')
def configure_labeling_budget(default_budget: int = 10) -> ConfigurationResult:
    """
    UI Component: Simple configuration to set the labeling budget.

    NOTE: This function body never executes. The decorator sends parameters
    to the UI, and the UI's response becomes the return value.

    Args:
        default_budget: Default labeling budget (sent to UI as default)

    Returns:
        Configuration dict with labeling_budget
        (returned from UI, not from this function body)
    """
    pass  # UI handles this - function body never executes


# =============================================================================
# DETECTION PHASE - BACKGROUND PROCESSING
# =============================================================================

@ui.component('function')
def setup_detection(
    dirty_path: str,
    clean_path: str,
    dataset_name: str,
    labeling_budget: int = 10,
    num_workers: Optional[int] = None,
    verbose: bool = True
) -> Tuple[DetectionParallel, dp.DatasetParallel, dict, Any]:
    """
    Initialize Dask cluster, load data, run strategies, generate features, and build clusters.

    This is a background processing step that prepares everything needed for
    interactive labeling.

    Args:
        dirty_path: Path to the dirty CSV file
        clean_path: Path to the clean CSV file (ground truth)
        dataset_name: Name of the dataset
        labeling_budget: Number of tuples to label
        num_workers: Number of Dask workers (defaults to CPU count)
        verbose: Enable verbose logging

    Returns:
        Tuple of (DetectionParallel instance, DatasetParallel instance, differences_dict, Dask client)
    """
    if num_workers is None:
        num_workers = os.cpu_count()

    # Create dataset dictionary
    dataset_dict = {
        "name": dataset_name,
        "path": dirty_path,
        "clean_path": clean_path
    }

    # Initialize detection application
    app = DetectionParallel()
    app.LABELING_BUDGET = labeling_budget
    app.VERBOSE = verbose

    # Initialize DataFrame in shared memory
    shared_df = app.initialize_dataframe(dirty_path)

    if verbose:
        print("Starting Dask cluster...")

    # Start Dask cluster
    client = app.start_dask_cluster(
        num_workers=num_workers,
        logging_level=logging.ERROR
    )
    client.run(app.init_workers)

    if verbose:
        print("Dask cluster started successfully.")

    # Initialize dataset
    dataset, differences_dict = app.initialize_dataset(dataset_dict)

    if verbose:
        print("Running error detection strategies...")

    # Run strategies
    app.run_strategies(dataset)

    if verbose:
        print("Generating features...")

    # Generate features
    app.generate_features(dataset)

    if verbose:
        print("Building clusters...")

    # Build clusters
    app.build_clusters(dataset)

    if verbose:
        print("Detection setup complete. Ready for interactive labeling.")

    return app, dataset, differences_dict, client


# =============================================================================
# DETECTION PHASE - INTERACTIVE LABELING
# =============================================================================

@ui.component('function')
def prepare_detection_samples(
    app: DetectionParallel,
    dataset: dp.DatasetParallel,
    budget: int
) -> List[Dict[str, Any]]:
    """
    PREPARE step: Sample tuples and format for labeling UI.

    This function samples tuples from the dataset and prepares them
    in a format suitable for the labeling UI component.

    Args:
        app: DetectionParallel instance
        dataset: DatasetParallel instance
        budget: Number of samples to prepare

    Returns:
        List of sample dictionaries for the UI
    """
    samples = []
    dataframe = container.shared_dataframe.read()

    for sample_idx in range(budget):
        # Sample a tuple
        app.sample_tuple(dataset)

        row_idx = dataset.sampled_tuple
        row_data = dataframe.iloc[row_idx]

        # Format for UI
        sample = {
            "id": int(row_idx),
            "sample_number": sample_idx + 1,
            "cells": []
        }

        # Add each cell in the row
        for col_idx, col_name in enumerate(dataframe.columns):
            cell_value = str(row_data[col_name])
            sample["cells"].append({
                "column": col_name,
                "column_index": col_idx,
                "value": cell_value
            })

        samples.append(sample)

    return samples


@ui.component('function')
def apply_detection_labels(
    dataset: dp.DatasetParallel,
    samples: List[Dict[str, Any]],
    labels: List[Dict[str, Any]],
    clean_dataframe: pandas.DataFrame
) -> None:
    """
    PROCESS step: Apply user labels from the UI to the dataset.

    This function takes the labels returned from the UI and applies them
    to the dataset's labeled_tuples and labeled_cells structures.

    Args:
        dataset: DatasetParallel instance
        samples: Original samples that were shown to user
        labels: User-provided labels from the UI
        clean_dataframe: Clean dataframe for corrections
    """
    # Create a lookup of labels by row_id
    label_lookup = {label["row_id"]: label for label in labels}

    for sample in samples:
        row_idx = sample["id"]
        dataset.labeled_tuples[row_idx] = 1

        # Check if we have labels for this row
        if row_idx in label_lookup:
            user_label_info = label_lookup[row_idx]
            user_label = user_label_info.get("user_label", "correct")

            # Apply labels to all cells in the row
            for cell_info in sample["cells"]:
                col_idx = cell_info["column_index"]
                cell = (row_idx, col_idx)

                # Determine if this cell is marked as erroneous
                # If the whole row is marked as "wrong", we need more specific cell labels
                # For now, use the row-level label
                if user_label == "wrong":
                    # Cell is erroneous
                    error_label = 1
                else:
                    # Cell is correct
                    error_label = 0

                # Get the correct value from ground truth
                correct_value = clean_dataframe.iloc[row_idx, col_idx]
                dataset.labeled_cells[cell] = [error_label, correct_value]
        else:
            # No label provided, use ground truth labeling
            for col_idx in range(len(sample["cells"])):
                cell = (row_idx, col_idx)
                correct_value = clean_dataframe.iloc[row_idx, col_idx]
                dirty_value = sample["cells"][col_idx]["value"]
                error_label = 1 if dirty_value != correct_value else 0
                dataset.labeled_cells[cell] = [error_label, correct_value]


@ui.component('simple-labeling')
def label_detection_samples(
    budget: int,
    data: List[Dict[str, Any]]
) -> LabelingResult:
    """
    UI Component: Interactive labeling of sampled tuples for error detection.

    NOTE: This function body never executes. The decorator sends parameters
    to the UI, and the UI's response becomes the return value.

    The user reviews each sampled row and marks it as correct, wrong, or skipped.

    Args:
        budget: Number of rows to label (sent to UI)
        data: List of sample data records to show the user (sent to UI)

    Returns:
        LabelingResult with 'labels' key containing list of label results
        (returned from UI, not from this function body)
    """
    pass  # UI handles this - function body never executes


@ui.component('function')
def run_detection_labeling(
    app: DetectionParallel,
    dataset: dp.DatasetParallel,
    differences_dict: dict,
    use_ground_truth: bool = False
) -> Dict[Tuple[int, int], str]:
    """
    Run the interactive detection labeling workflow.

    Args:
        app: DetectionParallel instance
        dataset: DatasetParallel instance
        differences_dict: Dictionary of actual errors
        use_ground_truth: If True, skip UI and use ground truth labels

    Returns:
        Dictionary of detected cells
    """
    clean_dataframe = dp.DatasetParallel.read_csv_dataframe(dataset.clean_path)

    if use_ground_truth:
        # Use ground truth labeling (non-interactive)
        while len(dataset.labeled_tuples) < app.LABELING_BUDGET:
            app.sample_tuple(dataset)
            app.label_with_ground_truth(dataset, differences_dict, clean_dataframe)
    else:
        # PREPARE: Sample tuples and format for UI
        samples = prepare_detection_samples(app, dataset, app.LABELING_BUDGET)

        # Format data for simple-labeling component
        ui_data = []
        for sample in samples:
            # Combine cell values into a display format
            row_text = " | ".join([
                f"{cell['column']}: {cell['value']}"
                for cell in sample["cells"]
            ])
            ui_data.append({
                "id": sample["id"],
                "name": f"Row {sample['id']}",
                "email": row_text  # Using email field for row data display
            })

        # DISPLAY: Get labels from user via UI
        result = label_detection_samples(
            budget=app.LABELING_BUDGET,
            data=ui_data
        )

        # PROCESS: Apply labels to dataset
        labels = result.get("labels", [])
        apply_detection_labels(dataset, samples, labels, clean_dataframe)

    # Propagate labels
    app.propagate_labels(dataset)

    # Predict labels
    app.predict_labels(dataset)

    return dataset.detected_cells


# =============================================================================
# DETECTION PHASE - RESULTS
# =============================================================================

@ui.component('function')
def format_detection_results(
    detected_cells: Dict[Tuple[int, int], str],
    dataset: dp.DatasetParallel
) -> List[Dict[str, Any]]:
    """
    PREPARE step: Format detection results for display in the UI.

    Args:
        detected_cells: Dictionary of detected error cells
        dataset: DatasetParallel instance for metrics

    Returns:
        List of result dictionaries formatted for the UI
    """
    metrics = dataset.get_data_cleaning_evaluation(detected_cells)[:3]
    precision, recall, f1 = metrics

    return [
        {"metric": "Detected Cells", "value": str(len(detected_cells))},
        {"metric": "Precision", "value": f"{precision:.2%}"},
        {"metric": "Recall", "value": f"{recall:.2%}"},
        {"metric": "F1 Score", "value": f"{f1:.2%}"},
    ]


@ui.component('simple-results')
def show_detection_results(
    results: List[Dict[str, Any]],
    title: str = "Detection Results"
) -> None:
    """
    UI Component: Display error detection results.

    NOTE: This function body never executes. The decorator sends parameters
    to the UI, which displays them and waits for user acknowledgment.

    Args:
        results: List of result dictionaries with metric/value pairs (sent to UI)
        title: Title for the results page (sent to UI)

    Returns:
        None - UI displays results and blocks until user clicks 'Continue'
    """
    pass  # UI handles this - function body never executes


@ui.component('function')
def finalize_detection(
    app: DetectionParallel,
    dataset: dp.DatasetParallel,
    client: Any,
    cleanup: bool = True
) -> Tuple[Dict[Tuple[int, int], str], Tuple[float, float, float]]:
    """
    Finalize detection phase and compute evaluation metrics.

    Args:
        app: DetectionParallel instance
        dataset: DatasetParallel instance
        client: Dask client
        cleanup: Whether to cleanup shared memory

    Returns:
        Tuple of (detected_cells dict, (precision, recall, f1))
    """
    detected_cells = dataset.detected_cells

    # Evaluate detection
    metrics = dataset.get_data_cleaning_evaluation(detected_cells)[:3]
    precision, recall, f1 = metrics

    if app.VERBOSE:
        print(f"\nDetection Results:")
        print(f"  Detected {len(detected_cells)} cells")
        print(f"  Precision: {precision:.2%}")
        print(f"  Recall: {recall:.2%}")
        print(f"  F1 Score: {f1:.2%}")

    # Store results
    if app.SAVE_RESULTS:
        app.store_results(dataset)

    # Cleanup if requested
    if cleanup:
        app.cleanup_raha(dataset)
        try:
            client.shutdown()
        except Exception:
            pass

    return detected_cells, (precision, recall, f1)


# =============================================================================
# CORRECTION PHASE - BACKGROUND PROCESSING
# =============================================================================

@ui.component('function')
def setup_correction(
    dirty_path: str,
    clean_path: str,
    dataset_name: str,
    detected_cells: Dict[Tuple[int, int], str],
    labeling_budget: int = 10,
    num_workers: Optional[int] = None,
    verbose: bool = True
) -> Tuple[CorrectionParallel, dp.DatasetParallel, Any]:
    """
    Initialize correction phase with detected cells from Raha.

    Args:
        dirty_path: Path to dirty CSV file
        clean_path: Path to clean CSV file
        dataset_name: Name of the dataset
        detected_cells: Dictionary of detected error cells from detection phase
        labeling_budget: Number of tuples to label for correction
        num_workers: Number of Dask workers
        verbose: Enable verbose logging

    Returns:
        Tuple of (CorrectionParallel instance, DatasetParallel instance, Dask client)
    """
    if num_workers is None:
        num_workers = os.cpu_count()

    dataset_dict = {
        "name": dataset_name,
        "path": dirty_path,
        "clean_path": clean_path
    }

    # Create a new dataset object with detected cells
    dataset = dp.DatasetParallel(dataset_dict)
    dataset.detected_cells = detected_cells

    # Initialize correction application
    app = CorrectionParallel()
    app.LABELING_BUDGET = labeling_budget
    app.VERBOSE = verbose

    # Initialize dataframes
    shared_df, clean_df = app.initialize_dataframes(dataset_dict)

    if verbose:
        print("Starting Dask cluster for correction...")

    # Start Dask cluster
    client = app.start_dask_cluster(
        num_workers=num_workers,
        logging_level=logging.ERROR
    )

    # Initialize dataset for correction
    dataset = app.initialize_dataset(dataset)

    # Initialize Dask task distribution
    column_workers = app.initialize_dask(dataset.column_errors)
    dataset.column_workers = column_workers

    # Initialize models
    app.initialize_models(dataset)

    if verbose:
        print("Correction setup complete. Ready for interactive labeling.")

    return app, dataset, client


# =============================================================================
# CORRECTION PHASE - INTERACTIVE LABELING
# =============================================================================

@ui.component('function')
def prepare_correction_sample(
    app: CorrectionParallel,
    dataset: dp.DatasetParallel
) -> Dict[str, Any]:
    """
    PREPARE step: Sample a tuple and format for correction labeling UI.

    This function samples a tuple from the dataset and prepares it
    in a format suitable for the correction labeling UI component.

    Args:
        app: CorrectionParallel instance
        dataset: DatasetParallel instance

    Returns:
        Sample dictionary for the UI
    """
    dataframe = container.shared_dataframe.read()

    # Sample a tuple
    app.sample_tuple(dataset)
    row_idx = dataset.sampled_tuple
    row_data = dataframe.iloc[row_idx]

    sample = {
        "id": int(row_idx),
        "cells": []
    }

    # Add cells, highlighting detected errors
    for col_idx, col_name in enumerate(dataframe.columns):
        cell = (row_idx, col_idx)
        cell_value = str(row_data[col_name])
        is_error = cell in dataset.detected_cells

        sample["cells"].append({
            "column": col_name,
            "column_index": col_idx,
            "value": cell_value,
            "is_detected_error": is_error
        })

    return sample


@ui.component('csv_cell_labeler')
def label_correction_cell(
    csv_path: str,
    row: int,
    column: str,
    context_rows: int = 3
) -> CellLabelingResult:
    """
    UI Component: Interactive cell-level labeling for error correction.

    NOTE: This function body never executes. The decorator sends parameters
    to the UI, and the UI's response becomes the return value.

    Shows a CSV table with a highlighted cell, and the user provides
    the correction for that cell.

    Args:
        csv_path: Path to the CSV file (sent to UI)
        row: Row index of the cell to correct (sent to UI)
        column: Column name of the cell to correct (sent to UI)
        context_rows: Number of surrounding rows to show for context (sent to UI)

    Returns:
        CellLabelingResult with is_correct and optional correction value
        (returned from UI, not from this function body)
    """
    pass  # UI handles this - function body never executes


@ui.component('function')
def run_correction_labeling(
    app: CorrectionParallel,
    dataset: dp.DatasetParallel,
    client: Any,
    use_ground_truth: bool = False
) -> Dict[Tuple[int, int], str]:
    """
    Run the interactive correction labeling workflow.

    Args:
        app: CorrectionParallel instance
        dataset: DatasetParallel instance
        client: Dask client
        use_ground_truth: If True, skip UI and use ground truth labels

    Returns:
        Dictionary of corrected cells
    """
    import multiprocessing.shared_memory as sm

    dataframe = container.shared_dataframe.read()
    clean_dataframe = container.shared_clean_dataframe.read()

    # Create shared memory object for dataset
    try:
        existing_mem = sm.SharedMemory(name="holy_dataset", create=False)
        existing_mem.close()
        existing_mem.unlink()
    except FileNotFoundError:
        pass

    dp.DatasetParallel.create_shared_object(dataset, "holy_dataset")

    # Reset labeled tuples for correction phase
    dataset.labeled_tuples = {}

    step = 0
    while len(dataset.labeled_tuples) < app.LABELING_BUDGET:
        start_time = time.time()

        # Sample a tuple
        app.sample_tuple(dataset)

        if use_ground_truth:
            # Use ground truth for labeling
            app.label_with_ground_truth(dataset, dataframe, clean_dataframe)
        else:
            # Interactive labeling would go here
            # For now, fall back to ground truth
            app.label_with_ground_truth(dataset, dataframe, clean_dataframe)

        # Update models
        app.update_models(dataset)

        # Initialize workers with new data
        client.run(
            app.initialize_workers,
            correct_instance=app,
            dataset_ref="holy_dataset",
            sampled_tuple=dataset.sampled_tuple,
            step=step
        )

        # Generate features and predict corrections
        dataset.column_prediction_futures = app.generate_and_predict(
            dataset.column_workers,
            dataset.column_errors,
            step
        )
        app.predict_corrections(dataset)

        step += 1
        end_time = time.time()

        if app.VERBOSE:
            print(f"Correction step {step}: {end_time - start_time:.2f}s")

    return dataset.corrected_cells


# =============================================================================
# CORRECTION PHASE - RESULTS
# =============================================================================

@ui.component('function')
def format_correction_results(
    corrected_cells: Dict[Tuple[int, int], str],
    dataset: dp.DatasetParallel
) -> List[Dict[str, Any]]:
    """
    PREPARE step: Format correction results for display in the UI.

    Args:
        corrected_cells: Dictionary of corrected cells
        dataset: DatasetParallel instance for metrics

    Returns:
        List of result dictionaries formatted for the UI
    """
    metrics = dataset.get_data_cleaning_evaluation(corrected_cells)[-3:]
    precision, recall, f1 = metrics

    return [
        {"metric": "Corrected Cells", "value": str(len(corrected_cells))},
        {"metric": "Precision", "value": f"{precision:.2%}"},
        {"metric": "Recall", "value": f"{recall:.2%}"},
        {"metric": "F1 Score", "value": f"{f1:.2%}"},
    ]


@ui.component('simple-results')
def show_correction_results(
    results: List[Dict[str, Any]],
    title: str = "Correction Results"
) -> None:
    """
    UI Component: Display error correction results.

    NOTE: This function body never executes. The decorator sends parameters
    to the UI, which displays them and waits for user acknowledgment.

    Args:
        results: List of result dictionaries with metric/value pairs (sent to UI)
        title: Title for the results page (sent to UI)

    Returns:
        None - UI displays results and blocks until user clicks 'Continue'
    """
    pass  # UI handles this - function body never executes


@ui.component('function')
def finalize_correction(
    app: CorrectionParallel,
    dataset: dp.DatasetParallel,
    client: Any,
    cleanup: bool = True
) -> Tuple[Dict[Tuple[int, int], str], Tuple[float, float, float]]:
    """
    Finalize correction phase and compute evaluation metrics.

    Args:
        app: CorrectionParallel instance
        dataset: DatasetParallel instance
        client: Dask client
        cleanup: Whether to cleanup shared memory

    Returns:
        Tuple of (corrected_cells dict, (precision, recall, f1))
    """
    corrected_cells = dataset.corrected_cells

    # Evaluate correction (uses last 3 metrics: ec_p, ec_r, ec_f)
    metrics = dataset.get_data_cleaning_evaluation(corrected_cells)[-3:]
    precision, recall, f1 = metrics

    if app.VERBOSE:
        print(f"\nCorrection Results:")
        print(f"  Corrected {len(corrected_cells)} cells")
        print(f"  Precision: {precision:.2%}")
        print(f"  Recall: {recall:.2%}")
        print(f"  F1 Score: {f1:.2%}")

    # Store results
    if app.SAVE_RESULTS:
        app.store_results(dataset)

    # Cleanup if requested
    if cleanup:
        app.cleanup_baran()
        try:
            client.shutdown()
        except Exception:
            pass

    return corrected_cells, (precision, recall, f1)


# =============================================================================
# MAIN ORCHESTRATOR
# =============================================================================

@ui.component('function')
def run_full_pipeline(
    dirty_path: str,
    clean_path: str,
    dataset_name: str,
    detection_budget: int = 10,
    correction_budget: int = 10,
    num_workers: Optional[int] = None,
    verbose: bool = True,
    use_ground_truth: bool = False
) -> Dict[str, Any]:
    """
    Run the complete data cleaning pipeline with Raha and Baran.

    Args:
        dirty_path: Path to dirty CSV file
        clean_path: Path to clean CSV file (ground truth)
        dataset_name: Name of the dataset
        detection_budget: Labeling budget for detection phase
        correction_budget: Labeling budget for correction phase
        num_workers: Number of Dask workers (defaults to CPU count)
        verbose: Enable verbose logging
        use_ground_truth: Skip UI and use ground truth for labeling

    Returns:
        Dictionary containing:
        - detected_cells: Dict of detected error cells
        - corrected_cells: Dict of corrected cells
        - detection_metrics: (precision, recall, f1) for detection
        - correction_metrics: (precision, recall, f1) for correction
    """
    if verbose:
        print("=" * 60)
        print("DaskRahaBaran Data Cleaning Pipeline")
        print("=" * 60)
        print(f"\nDataset: {dataset_name}")
        print(f"Dirty file: {dirty_path}")
        print(f"Clean file: {clean_path}")
        print(f"Detection budget: {detection_budget}")
        print(f"Correction budget: {correction_budget}")
        print()

    # =========================================================================
    # PHASE 1: ERROR DETECTION (Raha)
    # =========================================================================
    if verbose:
        print("-" * 60)
        print("PHASE 1: Error Detection (Raha)")
        print("-" * 60)

    # Setup detection
    det_app, det_dataset, differences_dict, det_client = setup_detection(
        dirty_path=dirty_path,
        clean_path=clean_path,
        dataset_name=dataset_name,
        labeling_budget=detection_budget,
        num_workers=num_workers,
        verbose=verbose
    )

    # Run detection labeling
    detected_cells = run_detection_labeling(
        app=det_app,
        dataset=det_dataset,
        differences_dict=differences_dict,
        use_ground_truth=use_ground_truth
    )

    # Finalize detection
    detected_cells, detection_metrics = finalize_detection(
        app=det_app,
        dataset=det_dataset,
        client=det_client,
        cleanup=True
    )

    # Show detection results via UI
    if not use_ground_truth:
        detection_results = [
            {"metric": "Detected Cells", "value": str(len(detected_cells))},
            {"metric": "Precision", "value": f"{detection_metrics[0]:.2%}"},
            {"metric": "Recall", "value": f"{detection_metrics[1]:.2%}"},
            {"metric": "F1 Score", "value": f"{detection_metrics[2]:.2%}"},
        ]
        show_detection_results(results=detection_results, title="Detection Results")

    # =========================================================================
    # PHASE 2: ERROR CORRECTION (Baran)
    # =========================================================================
    if verbose:
        print()
        print("-" * 60)
        print("PHASE 2: Error Correction (Baran)")
        print("-" * 60)

    # Setup correction
    cor_app, cor_dataset, cor_client = setup_correction(
        dirty_path=dirty_path,
        clean_path=clean_path,
        dataset_name=dataset_name,
        detected_cells=detected_cells,
        labeling_budget=correction_budget,
        num_workers=num_workers,
        verbose=verbose
    )

    # Run correction labeling
    corrected_cells = run_correction_labeling(
        app=cor_app,
        dataset=cor_dataset,
        client=cor_client,
        use_ground_truth=use_ground_truth
    )

    # Finalize correction
    corrected_cells, correction_metrics = finalize_correction(
        app=cor_app,
        dataset=cor_dataset,
        client=cor_client,
        cleanup=True
    )

    # Show correction results via UI
    if not use_ground_truth:
        correction_results = [
            {"metric": "Corrected Cells", "value": str(len(corrected_cells))},
            {"metric": "Precision", "value": f"{correction_metrics[0]:.2%}"},
            {"metric": "Recall", "value": f"{correction_metrics[1]:.2%}"},
            {"metric": "F1 Score", "value": f"{correction_metrics[2]:.2%}"},
        ]
        show_correction_results(results=correction_results, title="Correction Results")

    # =========================================================================
    # FINAL SUMMARY
    # =========================================================================
    if verbose:
        print()
        print("=" * 60)
        print("Pipeline Complete!")
        print("=" * 60)
        print(f"\nDetection: P={detection_metrics[0]:.2%}, R={detection_metrics[1]:.2%}, F1={detection_metrics[2]:.2%}")
        print(f"Correction: P={correction_metrics[0]:.2%}, R={correction_metrics[1]:.2%}, F1={correction_metrics[2]:.2%}")

    return {
        "detected_cells": detected_cells,
        "corrected_cells": corrected_cells,
        "detection_metrics": detection_metrics,
        "correction_metrics": correction_metrics
    }


def main():
    """
    Main entry point for the waschanlage pipeline.

    This can be run directly or the individual components can be
    imported and used separately.
    """
    # Get labeling budget from user via UI
    config = configure_labeling_budget(default_budget=10)
    labeling_budget = config["labeling_budget"]

    # Default to beers dataset for demonstration
    # Beers is a smaller dataset suitable for less powerful machines
    datasets_dir = Path(__file__).parent.parent / "datasets"

    result = run_full_pipeline(
        dirty_path=str(datasets_dir / "beers" / "dirty.csv"),
        clean_path=str(datasets_dir / "beers" / "clean.csv"),
        dataset_name="beers",
        detection_budget=labeling_budget,
        correction_budget=labeling_budget,
        verbose=True,
        use_ground_truth=False  # Set to False for interactive UI mode
    )

    return result


if __name__ == "__main__":
    main()