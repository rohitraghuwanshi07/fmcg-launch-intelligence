"""Generate the SIMULATED demo dataset (clearly labelled in dataset_meta.json)."""
import _bootstrap  # noqa
from src import demo_data

if __name__ == "__main__":
    print(demo_data.generate_and_save())
