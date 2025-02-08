import safetensors.torch
from safetensors.torch import load_file

model_path = "train_output/20250208213817/model.safetensors"  # Replace with the actual path
# model_path = "train_output/20250208214259/model.safetensors"
# model_path = "train_output/20250208214258/model-00001-of-00002.safetensors"

try:
    # Load the safetensors file
    data = load_file(model_path)

    # Print the keys (tensor names) inside the safetensors file
    print("Keys (Tensor Names) in the Safetensors File:")
    for key in data.keys():
        print(key, type(data[key]), data[key].shape)

    # Example:  Load a specific tensor (replace 'transformer.wte.weight' with an actual key)
    # specific_tensor_key = 'transformer.wte.weight'
    # if specific_tensor_key in data:
    #     tensor = data[specific_tensor_key]
    #     print(f"\nShape of tensor '{specific_tensor_key}': {tensor.shape}")
    #     print(f"Data type of tensor '{specific_tensor_key}': {tensor.dtype}")
    # else:
    #     print(f"Tensor '{specific_tensor_key}' not found in the safetensors file.")

except FileNotFoundError:
    print(f"Error: Safetensors file not found at '{model_path}'")
except Exception as e:
    print(f"Error loading safetensors file: {e}")