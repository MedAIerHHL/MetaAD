from .pet_dataset_2 import PETDataset

def get_dataset(dataset_name, **kwargs):
    if dataset_name == "PET":
        return PETDataset(**kwargs)
    else:
        raise ValueError(f"Dataset {dataset_name} not found.")

