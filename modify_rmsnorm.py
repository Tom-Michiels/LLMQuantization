#!/usr/bin/env python3
"""
Modify GGUF models to round RMSNorm fp32 weights to fp16.
"""

import os
import sys
import struct
import numpy as np
from pathlib import Path
from typing import List, Tuple
import gguf


def find_rmsnorm_tensors(reader: gguf.GGUFReader) -> List[str]:
    """Find all RMSNorm weight tensors in the model."""
    rmsnorm_tensors = []

    print("\nScanning for RMSNorm tensors...")
    for tensor in reader.tensors:
        # RMSNorm weights typically have these patterns in their names
        name_lower = tensor.name.lower()
        if any(pattern in name_lower for pattern in [
            "attention_norm.weight",
            "ffn_norm.weight",
            "norm.weight",
            ".attn_norm.",
            ".ffn_norm."
        ]):
            rmsnorm_tensors.append(tensor.name)
            print(f"  Found: {tensor.name}")
            print(f"    Shape: {tensor.shape}")
            print(f"    Type: {tensor.tensor_type}")

    return rmsnorm_tensors


def modify_gguf_rmsnorm(input_path: str, output_path: str) -> int:
    """
    Modify a GGUF file to round all RMSNorm weights from fp32 to fp16.
    Returns the number of tensors modified.
    """
    print(f"\n{'='*80}")
    print(f"Modifying: {input_path}")
    print(f"Output: {output_path}")
    print(f"{'='*80}")

    # Read the input file
    print("Reading GGUF file...")
    reader = gguf.GGUFReader(input_path)

    # Find RMSNorm tensors
    rmsnorm_names = find_rmsnorm_tensors(reader)

    if not rmsnorm_names:
        print("\n⚠ Warning: No RMSNorm tensors found!")
        print("This might mean the tensor naming convention is different.")
        print("\nAll tensor names in the model:")
        for tensor in reader.tensors:
            print(f"  {tensor.name}")
        return 0

    print(f"\nFound {len(rmsnorm_names)} RMSNorm tensors to modify")

    # Get architecture
    arch = "llama"
    for key, field in reader.fields.items():
        if field.name == "general.architecture":
            arch = str(bytes(field.parts[field.data[0]]), encoding="utf-8", errors="ignore")
            break

    print(f"Architecture: {arch}")

    # Create writer
    print("\nCreating new GGUF file...")
    writer = gguf.GGUFWriter(output_path, arch=arch)

    # Copy metadata/fields
    print("Copying metadata...")
    for key, field in reader.fields.items():
        if field.types:
            # Handle different field types
            if field.types[0] == gguf.GGUFValueType.STRING:
                parts_list = [bytes(part) for part in field.parts]
                writer.add_string(field.name, parts_list[field.data[0]])
            elif field.types[0] == gguf.GGUFValueType.UINT32:
                writer.add_uint32(field.name, field.data[0])
            elif field.types[0] == gguf.GGUFValueType.INT32:
                writer.add_int32(field.name, field.data[0])
            elif field.types[0] == gguf.GGUFValueType.FLOAT32:
                writer.add_float32(field.name, field.data[0])
            elif field.types[0] == gguf.GGUFValueType.ARRAY:
                # Skip arrays for now, or handle specially
                pass

    # Copy and modify tensors
    print("\nProcessing tensors...")
    modified_count = 0
    rmsnorm_set = set(rmsnorm_names)

    for tensor in reader.tensors:
        if tensor.name in rmsnorm_set:
            print(f"\n  Modifying: {tensor.name}")

            # Check if it's fp32
            if tensor.tensor_type == gguf.GGMLQuantizationType.F32:
                # Load the fp32 data
                fp32_data = np.frombuffer(tensor.data, dtype=np.float32).copy()

                print(f"    Original dtype: fp32")
                print(f"    Shape: {fp32_data.shape}")
                print(f"    Sample values (first 5): {fp32_data[:5]}")

                # Round to fp16 and back
                fp16_data = fp32_data.astype(np.float16)
                fp32_rounded = fp16_data.astype(np.float32)

                print(f"    After fp16 rounding: {fp32_rounded[:5]}")
                print(f"    Max difference: {np.max(np.abs(fp32_data - fp32_rounded))}")

                # Add the modified tensor
                writer.add_tensor(tensor.name, fp32_rounded)
                modified_count += 1
            else:
                print(f"    Skipping - not fp32 (type: {tensor.tensor_type})")
                # Copy as-is
                writer.add_tensor(tensor.name, tensor.data, raw_dtype=tensor.tensor_type)
        else:
            # Copy tensor as-is
            writer.add_tensor(tensor.name, tensor.data, raw_dtype=tensor.tensor_type)

    # Write the file
    print("\nWriting GGUF file...")
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()

    print(f"\n✓ Successfully modified {modified_count} tensors")
    print(f"✓ Output saved to: {output_path}")

    return modified_count


def main():
    """Main function to modify all downloaded models."""
    models_dir = Path("models")
    modified_dir = models_dir / "modified"
    modified_dir.mkdir(parents=True, exist_ok=True)

    print("="*80)
    print("Modifying RMSNorm Weights: fp32 → fp16 → fp32")
    print("="*80)

    # Find all GGUF files
    gguf_files = list(models_dir.glob("*.gguf"))

    if not gguf_files:
        print("No GGUF files found in models/ directory")
        print("Please run download_models.py first")
        return

    print(f"\nFound {len(gguf_files)} model(s) to process")

    results = {}
    for model_path in gguf_files:
        try:
            output_path = modified_dir / f"modified_{model_path.name}"
            count = modify_gguf_rmsnorm(str(model_path), str(output_path))
            results[model_path.name] = {
                "status": "success",
                "modified_tensors": count,
                "output": str(output_path)
            }
        except Exception as e:
            print(f"\n✗ Error processing {model_path.name}: {e}")
            import traceback
            traceback.print_exc()
            results[model_path.name] = {
                "status": "failed",
                "error": str(e)
            }

    # Print summary
    print("\n" + "="*80)
    print("Modification Summary")
    print("="*80)
    for filename, result in results.items():
        status = "✓" if result["status"] == "success" else "✗"
        print(f"{status} {filename}: {result.get('modified_tensors', 0)} tensors modified")

    return results


if __name__ == "__main__":
    main()
