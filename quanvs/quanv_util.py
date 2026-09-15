import sys
sys.path.append(".")

import constants
import math
import numpy as np
import torch
import torch.nn.functional as F


def default_encoding_config(encoding_approach):
    """
    Returns the default encoding configuration based on the specified encoding approach.
    """
    if encoding_approach == constants.CircuitEncoding.ROTATIONAL:
        return {
            "kernel_size": 3,
            "n_qubits": 9,
            "probability": 0.15,
            "n_shots": 1000,
            "activation": constants.QuanvActivation.HALF.value,
        }

    elif encoding_approach == constants.CircuitEncoding.INTEGRATED:
        return {
            "kernel_size": 3,
            "n_qubits": 4,
            "L": 25,
            "activation": constants.QuanvActivation.FULL.value,
            "n_shots": 1000,
        }

    elif encoding_approach == constants.CircuitEncoding.PASS:
        return {
            "info": "No encoding needed. This is a pass-through layer."
        }
    else:
        raise Exception(f"Encoding approach '{encoding_approach}' not recognized.")


def check_config_integrity(encoding_approach, encoding_config, kernel_size, verbose=False):
    if encoding_approach == constants.CircuitEncoding.ROTATIONAL:
        n_qubits = encoding_config["n_qubits"]
        if n_qubits != kernel_size ** 2:
            raise Exception(
                f"n_qubits must be equal to kernel_size^2 = {kernel_size**2} "
                "for rotational encoding."
            )

    default_config = default_encoding_config(encoding_approach)
    for key in default_config:
        if key not in encoding_config:
            raise Exception(
                f"Key '{key}' missing in encoding config for encoding approach "
                f"'{encoding_approach}'."
            )

    # Keep the original strict configuration contract.
    for key in encoding_config:
        if key not in default_config:
            raise Exception(
                f"Key '{key}' not recognized in encoding config for encoding "
                f"approach '{encoding_approach}'."
            )

    if verbose:
        print(f"Encoding approach: {encoding_approach}")
        print(f"Encoding config: {encoding_config}")
        print(f"Kernel size: {kernel_size}")
        print("Config integrity check passed.")


def check_existing_encoding(encoding_approach):
    if encoding_approach not in constants.CircuitEncoding:
        raise Exception(f"Encoding approach '{encoding_approach}' not recognized.")


def quantize_index(value, levels):
    """
    Return the integer quantization index used by the original implementation.

    The original project computes:
        min(1, floor(x * N) / (N - 1))

    Therefore the corresponding integer level is:
        min(N - 1, floor(x * N))
    """
    if levels is None:
        raise ValueError("levels cannot be None when requesting a quantization index.")
    if int(levels) < 2:
        raise ValueError("levels must be at least 2.")

    value = float(value)
    if value < 0 or value > 1:
        raise Exception(f"Value {value} is not in the range [0, 1].")

    return min(int(levels) - 1, math.floor(value * int(levels)))


def quantize(value, levels):
    """
    Assuming the input value is in [0, 1], quantize it using the original code's rule.
    """
    if value < 0 or value > 1:
        raise Exception(f"Value {value} is not in the range [0, 1].")
    levels = int(levels)
    if levels < 2:
        raise ValueError("levels must be at least 2.")

    index = quantize_index(value, levels)
    return index / (levels - 1)


def quantize_patch(patch, levels):
    """Quantize a patch in place, preserving the original function contract."""
    if levels is None:
        return patch

    kernel_size = len(patch)
    for ii in range(kernel_size ** 2):
        row = ii // kernel_size
        col = ii % kernel_size
        patch[row][col] = quantize(patch[row][col], levels)
    return patch


def quantize_patch_to_indices(patch, levels):
    """
    Quantize a patch and return both its floating-point representation and integer indices.
    """
    patch_array = np.asarray(patch, dtype=np.float32)
    levels = int(levels)
    if levels < 2:
        raise ValueError("levels must be at least 2.")

    indices = np.empty(patch_array.shape, dtype=np.int16)
    quantized = np.empty(patch_array.shape, dtype=np.float32)

    for index in np.ndindex(patch_array.shape):
        q_index = quantize_index(float(patch_array[index]), levels)
        indices[index] = q_index
        quantized[index] = q_index / (levels - 1)

    return quantized, indices


def adaptive_quantization_level(complexity, thresholds, levels=(10, 25, 50)):
    """
    Select a quantization level from a deterministic patch-complexity score.

    thresholds must contain two values: (low_to_medium, medium_to_high).
    """
    levels = tuple(sorted(int(x) for x in levels))
    if len(levels) != 3:
        raise ValueError("Exactly three quantization levels are required.")

    low_threshold, high_threshold = thresholds
    complexity = float(complexity)

    if complexity <= low_threshold:
        return levels[0]
    if complexity <= high_threshold:
        return levels[1]
    return levels[2]


def adaptive_quantize_patch(patch, levels, thresholds):
    """
    Quantize a patch using a deterministic level selected from local variance.

    Returns:
        quantized_patch, quantization_level, integer_indices, complexity
    """
    patch_array = np.asarray(patch, dtype=np.float32)
    complexity = float(np.var(patch_array))
    selected_levels = adaptive_quantization_level(
        complexity,
        thresholds,
        levels,
    )
    quantized, indices = quantize_patch_to_indices(
        patch_array,
        selected_levels,
    )
    return quantized, selected_levels, indices, complexity


def extract_patch_matrix(data, kernel_size, padding=0, stride=1):
    """
    Extract k x k patches using the same zero-padding convention as the quanvolutional layer.

    Input shape: [B, C, H, W].
    The current project uses one input channel, so the returned matrix contains
    flattened patches with k*k values.
    """
    if not isinstance(data, torch.Tensor):
        data = torch.as_tensor(data)
    if data.ndim != 4:
        raise ValueError("Expected data with shape [B, C, H, W].")
    if data.shape[1] != 1:
        raise ValueError("Training-patch analysis currently expects one input channel.")

    data = data.float()
    if padding > 0:
        data = F.pad(data, (padding, padding, padding, padding))

    patches = data.unfold(2, kernel_size, stride).unfold(3, kernel_size, stride)
    # [B, C, out_h, out_w, k, k] -> [num_patches, k*k]
    patches = patches[:, 0].contiguous()
    return patches.reshape(-1, kernel_size * kernel_size)


def analyze_training_patches(
    dataloader,
    kernel_size,
    padding=0,
    stride=1,
    max_calibration_patches=64,
    calibration_candidates_per_class=128,
):
    """
    Analyze training patches only and derive fixed, data-aware settings.

    The feature-importance score is deliberately classification-aware: it is
    a Fisher-style between-class/within-class ratio blended with global
    feature variance. This is used only to allocate the fixed L-gate budget.

    Calibration patches are selected deterministically and in a class-balanced,
    complexity-spread way so filter screening does not overfit only to the
    highest-variance patches. No test data is consumed.
    """
    feature_count = int(kernel_size ** 2)
    feature_sum = np.zeros(feature_count, dtype=np.float64)
    feature_sq_sum = np.zeros(feature_count, dtype=np.float64)
    total_patch_count = 0
    complexity_chunks = []

    class_counts = {}
    class_sums = {}
    class_sq_sums = {}
    candidates_by_class = {}

    for data, target in dataloader:
        patches = extract_patch_matrix(
            data,
            kernel_size=kernel_size,
            padding=padding,
            stride=stride,
        )

        if patches.shape[0] == 0 or len(target) == 0:
            continue

        batch_np = patches.cpu().numpy().astype(np.float32)
        patch_count = int(batch_np.shape[0])

        feature_sum += batch_np.sum(axis=0, dtype=np.float64)
        feature_sq_sum += np.square(batch_np).sum(axis=0, dtype=np.float64)
        total_patch_count += patch_count

        complexity = np.var(batch_np, axis=1).astype(np.float32)
        complexity_chunks.append(complexity)

        patches_per_image = patch_count // len(target)
        expanded_labels = (
            target.detach().cpu().numpy().reshape(-1).repeat(patches_per_image)
        )

        # Defensive alignment check: the unfold operation and label expansion
        # must describe exactly the same number of local patches.
        expanded_labels = expanded_labels[:patch_count]

        for label in np.unique(expanded_labels):
            label = int(label)
            mask = expanded_labels == label
            class_patches = batch_np[mask]
            if class_patches.shape[0] == 0:
                continue

            class_counts[label] = class_counts.get(label, 0) + class_patches.shape[0]
            class_sums[label] = class_sums.get(label, np.zeros(feature_count)) + class_patches.sum(axis=0, dtype=np.float64)
            class_sq_sums[label] = class_sq_sums.get(label, np.zeros(feature_count)) + np.square(class_patches).sum(axis=0, dtype=np.float64)

            # Keep only a small complexity-spread pool from each batch.
            # Storing every local patch would be unnecessarily memory-heavy
            # because an image contributes hundreds of overlapping patches.
            class_indices = np.flatnonzero(mask)
            if len(class_indices) <= 8:
                selected_local = class_indices
            else:
                order = class_indices[np.argsort(complexity[class_indices])]
                positions = np.linspace(0, len(order) - 1, 8, dtype=int)
                selected_local = order[positions]

            bucket = candidates_by_class.setdefault(label, [])
            for patch_index in selected_local:
                bucket.append(
                    (
                        float(complexity[patch_index]),
                        batch_np[patch_index].reshape(kernel_size, kernel_size).copy(),
                    )
                )

    if total_patch_count == 0:
        raise ValueError("No training patches were available for analysis.")

    mean = feature_sum / total_patch_count
    variance = np.maximum(
        feature_sq_sum / total_patch_count - np.square(mean),
        0.0,
    )

    # Classification-aware feature importance.
    # Between-class variance tells us how differently a pixel position behaves
    # across classes; within-class variance penalizes positions that are noisy
    # but not discriminative.
    between = np.zeros(feature_count, dtype=np.float64)
    within = np.zeros(feature_count, dtype=np.float64)
    for label, count in class_counts.items():
        count = int(count)
        if count <= 0:
            continue
        class_mean = class_sums[label] / count
        class_var = np.maximum(
            class_sq_sums[label] / count - np.square(class_mean),
            0.0,
        )
        between += count * np.square(class_mean - mean)
        within += count * class_var

    between /= max(total_patch_count, 1)
    within /= max(total_patch_count, 1)

    fisher = (between + 1e-8) / (within + 1e-8)

    # Normalize and blend with variance. The blend prevents the gate allocator
    # from collapsing onto only a few highly class-separating positions.
    fisher_norm = fisher / max(float(fisher.sum()), 1e-12)
    variance_norm = (variance + 1e-8) / max(float((variance + 1e-8).sum()), 1e-12)
    feature_importance = 0.75 * fisher_norm + 0.25 * variance_norm
    feature_importance /= max(float(feature_importance.sum()), 1e-12)

    all_complexity = np.concatenate(complexity_chunks)
    low_threshold, high_threshold = np.quantile(
        all_complexity,
        [0.40, 0.85],
    )
    if high_threshold <= low_threshold:
        high_threshold = low_threshold + 1e-8

    # Deterministic, class-balanced, complexity-spread calibration set.
    labels_sorted = sorted(candidates_by_class)
    class_quota = max(1, max_calibration_patches // max(1, len(labels_sorted)))
    calibration_records = []

    for label in labels_sorted:
        records = sorted(
            candidates_by_class[label],
            key=lambda item: item[0],
        )
        if not records:
            continue
        records = records[: max(calibration_candidates_per_class, class_quota)]

        take = min(class_quota, len(records))
        if take == 1:
            chosen_positions = [len(records) // 2]
        else:
            chosen_positions = np.linspace(
                0, len(records) - 1, take, dtype=int
            ).tolist()

        for position in chosen_positions:
            score, patch = records[position]
            calibration_records.append((score, patch, label))

    # Fill any remaining slots deterministically from under-represented classes
    # and then from the remaining records with the widest complexity coverage.
    if len(calibration_records) < max_calibration_patches:
        used_keys = {
            (label, float(score), tuple(patch.reshape(-1).tolist()))
            for score, patch, label in calibration_records
        }
        remainder = []
        for label in labels_sorted:
            for score, patch in candidates_by_class[label]:
                key = (label, float(score), tuple(patch.reshape(-1).tolist()))
                if key not in used_keys:
                    remainder.append((score, patch, label))
        remainder.sort(key=lambda item: (item[2], item[0]))
        calibration_records.extend(
            remainder[: max_calibration_patches - len(calibration_records)]
        )

    calibration_records = calibration_records[:max_calibration_patches]
    calibration_patches = [item[1] for item in calibration_records]
    calibration_labels = [item[2] for item in calibration_records]

    return {
        "feature_importance": feature_importance.astype(np.float64),
        "feature_variance": variance.astype(np.float64),
        "feature_between_class_variance": between.astype(np.float64),
        "feature_within_class_variance": within.astype(np.float64),
        "feature_fisher_score": fisher.astype(np.float64),
        "quantization_thresholds": (
            float(low_threshold),
            float(high_threshold),
        ),
        "calibration_patches": calibration_patches,
        "calibration_labels": calibration_labels,
        "num_training_patches": int(total_patch_count),
    }


def get_qc_characteristics(qc):
    depth = qc.depth()
    num_qubits = qc.num_qubits
    ops = {}
    for op in qc.data:
        gate_name = op.operation.name
        ops[gate_name] = ops.get(gate_name, 0) + 1
    num_multi_qubit_gates = qc.num_nonlocal_gates()
    return {
        "depth": depth,
        "num_qubits": num_qubits,
        "ops": ops,
        "num_multi_qubit_gates": num_multi_qubit_gates,
    }


def print_qc_characteristics(qc):
    characteristics = get_qc_characteristics(qc)
    print("Quantum circuit characteristics:")
    print(f"Depth: {characteristics['depth']}")
    print(f"Number of qubits: {characteristics['num_qubits']}")
    print("Operations:")
    for op in characteristics["ops"]:
        print(f"{op}: {characteristics['ops'][op]}")
    print(
        "Number of multi-qubit gates: "
        f"{characteristics['num_multi_qubit_gates']}"
    )
