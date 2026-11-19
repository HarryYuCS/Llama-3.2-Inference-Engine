"""

Load the .safetensors file and extract certain weights to inform WeightStore validation

"""

from safetensors import safe_open
from pathlib import Path

WEIGHTS = Path(__file__).resolve().parents[2] / "weights"

with safe_open(filename=str(WEIGHTS) + "/model.safetensors", framework="pt", device="cpu") as f:
    # inspect metadata
    metadata = f.metadata()
    print("Metadata:", metadata)

    print(f"\nTotal tensors: {len(f.keys())}")
    for key in f.keys():
        tensor_slice = f.get_slice(key)
        print(f"{key} | shape: {tensor_slice.get_shape()} | dtype: {tensor_slice.get_dtype()} | first element: {tensor_slice[:].flatten()[0]}")