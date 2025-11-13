#!/usr/bin/env python3
"""
Measure perplexity using llama.cpp for all models.
"""

import os
import sys
import json
import subprocess
from pathlib import Path
from typing import Dict, Optional
import re


def run_perplexity(model_path: str, test_file: str = "wikitext-2/wiki.test.raw") -> Optional[float]:
    """
    Run llama.cpp perplexity measurement and parse the result.
    """
    llamacpp_perplexity = "./llama.cpp/perplexity"

    if not os.path.exists(llamacpp_perplexity):
        print(f"Error: {llamacpp_perplexity} not found. Please run setup_llamacpp.sh first.")
        return None

    if not os.path.exists(test_file):
        print(f"Error: {test_file} not found. Please run setup_llamacpp.sh first.")
        return None

    print(f"\nMeasuring perplexity for: {model_path}")
    print(f"Using test file: {test_file}")

    try:
        # Run perplexity measurement
        cmd = [
            llamacpp_perplexity,
            "-m", model_path,
            "-f", test_file,
            "--perplexity",
            "-ngl", "0",  # CPU only for consistency
            "-t", str(os.cpu_count() or 4)
        ]

        print(f"Running: {' '.join(cmd)}")
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=1800  # 30 minute timeout
        )

        output = result.stdout + result.stderr

        # Parse perplexity from output
        # llama.cpp outputs something like "Final perplexity: 12.34"
        perplexity = None
        for line in output.split('\n'):
            # Look for perplexity value
            if 'perplexity' in line.lower():
                # Try to extract the number
                match = re.search(r'perplexity[:\s]+([0-9.]+)', line.lower())
                if match:
                    perplexity = float(match.group(1))
                    print(f"✓ Perplexity: {perplexity}")
                    break

        if perplexity is None:
            print("Warning: Could not parse perplexity from output")
            print("Output:")
            print(output)

        return perplexity

    except subprocess.TimeoutExpired:
        print("Error: Perplexity measurement timed out")
        return None
    except Exception as e:
        print(f"Error running perplexity: {e}")
        import traceback
        traceback.print_exc()
        return None


def measure_all_models(original_dir: str = "models", modified_dir: str = "models/modified") -> Dict:
    """
    Measure perplexity for all original and modified models.
    """
    results = {
        "original": {},
        "modified": {}
    }

    # Measure original models
    print("\n" + "="*80)
    print("Measuring Perplexity: Original Models")
    print("="*80)

    original_path = Path(original_dir)
    for model_file in sorted(original_path.glob("*.gguf")):
        perplexity = run_perplexity(str(model_file))
        if perplexity is not None:
            results["original"][model_file.name] = perplexity

    # Measure modified models
    print("\n" + "="*80)
    print("Measuring Perplexity: Modified Models (RMSNorm fp16)")
    print("="*80)

    modified_path = Path(modified_dir)
    if modified_path.exists():
        for model_file in sorted(modified_path.glob("*.gguf")):
            perplexity = run_perplexity(str(model_file))
            if perplexity is not None:
                results["modified"][model_file.name] = perplexity
    else:
        print(f"Modified directory not found: {modified_dir}")
        print("Please run modify_rmsnorm.py first")

    return results


def print_results_table(results: Dict):
    """Print results in a nice table format."""
    print("\n" + "="*80)
    print("PERPLEXITY RESULTS")
    print("="*80)

    # Match original and modified models
    print(f"\n{'Model':<30} {'Original':<15} {'Modified (fp16)':<15} {'Difference':<15}")
    print("-" * 80)

    for orig_name, orig_ppl in sorted(results["original"].items()):
        # Find corresponding modified model
        modified_name = f"modified_{orig_name}"
        modified_ppl = results["modified"].get(modified_name, None)

        if modified_ppl is not None:
            diff = modified_ppl - orig_ppl
            diff_str = f"+{diff:.4f}" if diff > 0 else f"{diff:.4f}"
            print(f"{orig_name:<30} {orig_ppl:<15.4f} {modified_ppl:<15.4f} {diff_str:<15}")
        else:
            print(f"{orig_name:<30} {orig_ppl:<15.4f} {'N/A':<15} {'N/A':<15}")

    # Print any modified models without originals
    for mod_name, mod_ppl in sorted(results["modified"].items()):
        orig_name = mod_name.replace("modified_", "")
        if orig_name not in results["original"]:
            print(f"{mod_name:<30} {'N/A':<15} {mod_ppl:<15.4f} {'N/A':<15}")


def main():
    """Main function."""
    print("="*80)
    print("LLM Quantization Perplexity Measurement")
    print("="*80)

    # Check if llama.cpp is set up
    if not os.path.exists("llama.cpp/perplexity"):
        print("\n✗ llama.cpp not found or not built")
        print("Please run: bash setup_llamacpp.sh")
        return

    # Measure all models
    results = measure_all_models()

    # Save results
    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)

    results_file = results_dir / "perplexity_results.json"
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n✓ Results saved to: {results_file}")

    # Print results table
    print_results_table(results)

    # Calculate statistics
    if results["original"] and results["modified"]:
        print("\n" + "="*80)
        print("SUMMARY")
        print("="*80)

        avg_diff = 0
        count = 0
        for orig_name, orig_ppl in results["original"].items():
            modified_name = f"modified_{orig_name}"
            if modified_name in results["modified"]:
                diff = results["modified"][modified_name] - orig_ppl
                avg_diff += diff
                count += 1

        if count > 0:
            avg_diff /= count
            print(f"Average perplexity change: {avg_diff:+.4f}")
            print(f"(Positive means perplexity increased with fp16 RMSNorm)")


if __name__ == "__main__":
    main()
