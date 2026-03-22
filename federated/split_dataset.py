import pandas as pd
from pathlib import Path
import os
import numpy as np

def split_and_save_dataset(dataset_name, output_dir, n_splits=2, random_state=42):
    """
    Splits dirty.csv and clean.csv into random subsets.
    Saves indices to .npy files to allow global mapping later.
    """
    base_path = Path(f"datasets/{dataset_name}")
    dirty_path = base_path / "dirty.csv"
    clean_path = base_path / "clean.csv"

    print(f"Loading {dataset_name} datasets...")
    df_dirty = pd.read_csv(dirty_path)
    df_clean = pd.read_csv(clean_path)

    indices = np.array(df_dirty.index.tolist())
    np.random.seed(random_state)
    np.random.shuffle(indices)

    split_indices_list = np.array_split(indices, n_splits)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Splitting {dataset_name} dataset into {n_splits} parts...")
    for i, split_indices in enumerate(split_indices_list):
        client_id = i + 1
        
        # Save subsets WITHOUT index column so Raha reads them normally
        df_dirty.loc[split_indices].to_csv(output_dir / f"dirty_{client_id}.csv", index=False)
        df_clean.loc[split_indices].to_csv(output_dir / f"clean_{client_id}.csv", index=False)
        
        # Save the mapping (Global ID for each local row)
        np.save(output_dir / f"indices_{client_id}.npy", split_indices)
        
        print(f"- Client {client_id} saved to {output_dir}")

if __name__ == "__main__":
    dataset_to_split = "hospital"
    n = 10
    output_directory = Path(f"datasets/{dataset_to_split}/split_{n}")
    split_and_save_dataset(dataset_to_split, output_directory, n_splits=n)
