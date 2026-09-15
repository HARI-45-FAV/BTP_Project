# Quanvolutional_Layer.py
import time
import math

import constants
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from qiskit.circuit import QuantumCircuit
from qiskit_ibm_runtime import SamplerV2
from qiskit_aer import AerSimulator

from quanvs import quanv_util as utils
import quanvs.circuit_builder as circuit_builder

from constants import QuantumDevice


class QuanvolutionalLayer(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        quantization=2,
        encoding_approach=None,
        encoding_config=None,
        simulator=QuantumDevice.NOISELESS,
        verbose=False,
        research_config=None,
        feature_importance=None,
        quantization_thresholds=None,
        filter_calibration_patches=None,
        filter_calibration_labels=None,
    ):
        super(QuanvolutionalLayer, self).__init__()

        if simulator == QuantumDevice.NOISELESS:
            self.simulator = AerSimulator(method="statevector")
            self.need_to_transpile = False
        else:
            raise Exception(f"Simulator '{simulator}' not recognized.")

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = int(kernel_size)
        self.stride = int(stride)
        self.padding = padding
        self.verbose = verbose

        assert padding % 0.5 == 0, "Padding must be a multiple of 0.5."
        self.int_padding = int(padding)
        self.partial_padding = self.padding - self.int_padding

        self.quantization = quantization
        self.encoding_approach = encoding_approach
        self.encoding_config = encoding_config

        utils.check_existing_encoding(encoding_approach)
        if encoding_config is None:
            encoding_config = utils.default_encoding_config(encoding_approach)
            self.encoding_config = encoding_config

        utils.check_config_integrity(
            encoding_approach,
            self.encoding_config,
            self.kernel_size,
            verbose,
        )

        # ---------------------------------------------------------
        # Research-method controls
        # ---------------------------------------------------------
        research_config = research_config or {}
        self.adaptive_quantization = bool(
            research_config.get("adaptive_quantization", True)
        )
        self.information_aware_gates = bool(
            research_config.get("information_aware_gates", True)
        )
        self.adaptive_levels = tuple(
            int(x) for x in research_config.get(
                "quantization_levels", (10, 25, 50)
            )
        )
        self.quantization_thresholds = quantization_thresholds

        self.similarity_memoization = bool(
            research_config.get("similarity_memoization", True)
        )
        self.similarity_threshold = float(
            research_config.get("similarity_threshold", 0.05)
        )
        self.similarity_coarse_levels = int(
            research_config.get("similarity_coarse_levels", 8)
        )
        self.max_prototypes_per_bucket = int(
            research_config.get("max_prototypes_per_bucket", 32)
        )
        self.similarity_validation_limit = int(
            research_config.get("similarity_validation_limit", 32)
        )
        self.similarity_output_tolerance = float(
            research_config.get("similarity_output_tolerance", 0.075)
        )

        self.adaptive_shots = bool(
            research_config.get("adaptive_shots", True)
        )
        self.shots_by_level = {
            int(k): int(v)
            for k, v in research_config.get(
                "shots_by_level", {10: 250, 25: 500, 50: 1000}
            ).items()
        }

        self.filter_selection_enabled = bool(
            research_config.get("filter_selection", True)
        )
        self.filter_candidates = max(
            self.out_channels,
            int(research_config.get("filter_candidates", 16)),
        )
        self.filter_screening_shots = int(
            research_config.get("filter_screening_shots", 100)
        )
        self.filter_quality_weight = float(
            research_config.get("filter_quality_weight", 0.40)
        )
        self.filter_diversity_weight = float(
            research_config.get("filter_diversity_weight", 0.60)
        )
        self.quantum_seed = int(research_config.get("quantum_seed", 2026))
        self.feature_importance = (
            feature_importance if self.information_aware_gates else None
        )
        self.filter_calibration_patches = filter_calibration_patches or []
        self.filter_calibration_labels = filter_calibration_labels or []

        # ---------------------------------------------------------
        # Memoization structures
        # ---------------------------------------------------------
        self.look_up = {}
        self.prototype_buckets = {}
        self.similarity_validation_count = 0

        self.counters = {
            "images": 0,
            "total_patches": 0,
            "unique_patches": 0,
            "quantum_executions": 0,
            "exact_cache_hits": 0,
            "similarity_cache_hits": 0,
            "similarity_validations": 0,
            "similarity_validation_failures": 0,
            "similarity_prototypes": 0,
            "screening_executions": 0,
            "shots_250": 0,
            "shots_500": 0,
            "shots_1000": 0,
        }

        self.filter_selection_summary = {}

        # Creating the sampler once avoids reconstructing the sampler object for
        # every single patch while preserving the same execution semantics.
        self.sampler = SamplerV2(mode=self.simulator)

        # ---------------------------------------------------------
        # Generate the fixed quantum filters once
        # ---------------------------------------------------------
        start_time = time.time()
        self.circuits = self.generate_layer()
        end_time = time.time()

        if verbose:
            print(
                "Time required to generate circuit layer: "
                f"{end_time - start_time:.4f} seconds"
            )

    # =============================================================
    # PATCH PREPARATION
    # =============================================================

    def _prepare_patch(self, patch):
        patch = np.asarray(patch, dtype=np.float32)

        if self.adaptive_quantization:
            if self.quantization_thresholds is None:
                raise ValueError(
                    "Adaptive quantization requires training-derived "
                    "quantization thresholds."
                )

            (
                quantized,
                levels,
                indices,
                complexity,
            ) = utils.adaptive_quantize_patch(
                patch,
                self.adaptive_levels,
                self.quantization_thresholds,
            )
        else:
            complexity = float(np.var(patch))
            if self.quantization is None:
                quantized = patch.copy()
                indices = None
                levels = None
            else:
                quantized, indices = utils.quantize_patch_to_indices(
                    patch,
                    int(self.quantization),
                )
                levels = int(self.quantization)

        shots = self._shots_for_patch(levels)
        return quantized, levels, indices, complexity, shots

    def _shots_for_patch(self, levels):
        config = self.encoding_config
        if config is None:
            config = utils.default_encoding_config(self.encoding_approach)
            self.encoding_config = config

        fallback_shots = int(config.get("n_shots", 1024))

        if not self.adaptive_shots or levels is None:
            return fallback_shots

        if int(levels) in self.shots_by_level:
            return int(self.shots_by_level[int(levels)])

        # Safe fallback: use the configured maximum shot count.
        return fallback_shots

    # =============================================================
    # SIMILARITY MEMOIZATION
    # =============================================================

    def _coarse_signature(self, indices, levels):
        values = np.asarray(indices, dtype=np.float32)
        if levels <= 1:
            normalized = values
        else:
            normalized = values / float(levels - 1)

        bins = self.similarity_coarse_levels
        mean_value = float(normalized.mean())
        std_value = float(normalized.std())
        energy_value = float(np.mean(np.square(normalized)))

        return (
            min(bins - 1, int(mean_value * bins)),
            min(bins - 1, int(std_value * bins)),
            min(bins - 1, int(energy_value * bins)),
        )

    def _neighbor_signatures(self, signature):
        bins = self.similarity_coarse_levels
        mean_bin, std_bin, energy_bin = signature

        for dm in (-1, 0, 1):
            for ds in (-1, 0, 1):
                for de in (-1, 0, 1):
                    yield (
                        min(bins - 1, max(0, mean_bin + dm)),
                        min(bins - 1, max(0, std_bin + ds)),
                        min(bins - 1, max(0, energy_bin + de)),
                    )

    def _find_similar_prototype(
        self,
        channel,
        levels,
        shots,
        indices,
    ):
        if not self.similarity_memoization or levels is None:
            return None

        signature = self._coarse_signature(indices, levels)
        best_value = None
        best_distance = float("inf")

        for neighbor in self._neighbor_signatures(signature):
            bucket_key = (channel, int(levels), int(shots), neighbor)
            bucket = self.prototype_buckets.get(bucket_key, [])

            for prototype_indices, prototype_value in bucket:
                prototype = np.asarray(
                    prototype_indices,
                    dtype=np.float32
                ).reshape(-1)

                current = np.asarray(
                    indices,
                    dtype=np.float32
                ).reshape(-1)
                distance = float(
                    np.sqrt(
                        np.mean(
                            np.square(
                                (current - prototype)
                                / max(1, levels - 1)
                            )
                        )
                    )
                )

                if distance <= self.similarity_threshold and distance < best_distance:
                    best_distance = distance
                    best_value = prototype_value

        return best_value

    def _add_similarity_prototype(
        self,
        channel,
        levels,
        shots,
        indices,
        value,
    ):
        if not self.similarity_memoization or levels is None:
            return

        signature = self._coarse_signature(indices, levels)
        bucket_key = (channel, int(levels), int(shots), signature)
        bucket = self.prototype_buckets.setdefault(bucket_key, [])

        if len(bucket) >= self.max_prototypes_per_bucket:
            return

        prototype_indices = tuple(
            int(x) for x in np.asarray(indices).reshape(-1)
        )
        bucket.append((prototype_indices, float(value)))
        self.counters["similarity_prototypes"] += 1

    # =============================================================
    # FORWARD
    # =============================================================

    def forward(self, x):
        if not isinstance(x, torch.Tensor):
            print(
                "Alert: Input is not a tensor. Converting to tensor."
            )
            x = torch.tensor(x)

        self.device = x.device

        if len(x.shape) == 2:
            x = x.unsqueeze(0).unsqueeze(0)
        elif len(x.shape) == 3:
            x = x.unsqueeze(0)

        batch_size, _, height, width = x.shape

        padded_height = int(height + 2 * self.padding)
        padded_width = int(width + 2 * self.padding)

        out_height = ((padded_height - self.kernel_size) // self.stride) + 1
        out_width = ((padded_width - self.kernel_size) // self.stride) + 1

        output = torch.zeros(
            (
                batch_size,
                self.out_channels,
                out_height,
                out_width,
            ),
            device=self.device,
        )

        start_time = time.time()

        for i in range(batch_size):
            image_start_time = time.time()
            self.counters["images"] += 1

            padded_x = F.pad(
                x[i],
                (
                    self.int_padding,
                    self.int_padding,
                    self.int_padding,
                    self.int_padding,
                ),
            )

            if self.partial_padding > 0:
                padded_x = F.pad(padded_x, (1, 0, 1, 0))

            for j in range(self.out_channels):
                for h in range(out_height):
                    for w in range(out_width):
                        h_start = h * self.stride
                        h_end = h_start + self.kernel_size
                        w_start = w * self.stride
                        w_end = w_start + self.kernel_size

                        patch = padded_x[
                            :,
                            h_start:h_end,
                            w_start:w_end,
                        ].cpu().numpy()[0]

                        (
                            patch,
                            levels,
                            indices,
                            _complexity,
                            shots,
                        ) = self._prepare_patch(patch)

                        patch_key = (
                            j,
                            levels,
                            shots,
                            tuple(patch.reshape(-1).tolist()),
                        )

                        # Keep the original dataset-level patch counter behavior:
                        # count each spatial patch once rather than once per filter.
                        if j == 0:
                            self.counters["total_patches"] += 1

                        value = self.look_up.get(patch_key)
                        if value is not None:
                            self.counters["exact_cache_hits"] += 1
                        else:
                            value = self._find_similar_prototype(
                                j,
                                levels,
                                shots,
                                indices,
                            )

                            if value is not None:
                                self.counters["similarity_cache_hits"] += 1

                                # Validate a bounded number of approximate reuses.
                                if (
                                    self.similarity_validation_count
                                    < self.similarity_validation_limit
                                ):
                                    exact_value = self._execute_quantum(
                                        self.circuits[j],
                                        patch,
                                        shots,
                                    )
                                    self.counters["quantum_executions"] += 1
                                    self._record_shot_count(shots)
                                    self.similarity_validation_count += 1
                                    self.counters["similarity_validations"] += 1

                                    error = abs(exact_value - value)
                                    if error > self.similarity_output_tolerance:
                                        self.counters[
                                            "similarity_validation_failures"
                                        ] += 1
                                        value = exact_value
                                        self.look_up[patch_key] = value
                                        self._add_similarity_prototype(
                                            j,
                                            levels,
                                            shots,
                                            indices,
                                            value,
                                        )
                                    else:
                                        self.look_up[patch_key] = value
                                else:
                                    self.look_up[patch_key] = value
                            else:
                                value = self._execute_quantum(
                                    self.circuits[j],
                                    patch,
                                    shots,
                                )
                                self.counters["quantum_executions"] += 1
                                self._record_shot_count(shots)
                                self.look_up[patch_key] = value
                                self._add_similarity_prototype(
                                    j,
                                    levels,
                                    shots,
                                    indices,
                                    value,
                                )
                                if j == 0:
                                    self.counters["unique_patches"] += 1

                        output[i, j, h, w] = value

            image_end_time = time.time()
            image_time = image_end_time - image_start_time
            elapsed_time = image_end_time - start_time
            average_time_per_image = elapsed_time / (i + 1)
            estimated_remaining_time = average_time_per_image * (
                batch_size - i - 1
            )

            if self.verbose and i % 5 == 4:
                print(
                    f"Time for image {i + 1}: "
                    f"{image_time:.2f} seconds"
                )
                print(
                    "Estimated remaining time: "
                    f"{estimated_remaining_time / 60:.2f} minutes"
                )
                print(self.counters)

        return output

    # =============================================================
    # SINGLE CHANNEL FORWARD
    # =============================================================

    def single_forward(self, x, circuit_index):
        batch_size, _, height, width = x.shape

        padded_height = height + 2 * self.padding
        padded_width = width + 2 * self.padding

        out_height = ((padded_height - self.kernel_size) // self.stride) + 1
        out_width = ((padded_width - self.kernel_size) // self.stride) + 1

        output = torch.zeros(
            (
                batch_size,
                1,
                out_height,
                out_width,
            ),
            device=x.device,
        )

        for i in range(batch_size):
            padded_x = F.pad(
                x[i],
                (
                    self.int_padding,
                    self.int_padding,
                    self.int_padding,
                    self.int_padding,
                ),
            )

            if self.partial_padding > 0:
                padded_x = F.pad(padded_x, (1, 0, 1, 0))

            for h in range(out_height):
                for w in range(out_width):
                    h_start = h * self.stride
                    h_end = h_start + self.kernel_size
                    w_start = w * self.stride
                    w_end = w_start + self.kernel_size

                    patch = padded_x[
                        :,
                        h_start:h_end,
                        w_start:w_end,
                    ].cpu().numpy()[0]

                    (
                        patch,
                        levels,
                        indices,
                        _complexity,
                        shots,
                    ) = self._prepare_patch(patch)

                    patch_key = (
                        circuit_index,
                        levels,
                        shots,
                        tuple(patch.reshape(-1).tolist()),
                    )

                    value = self.look_up.get(patch_key)
                    if value is None:
                        value = self._find_similar_prototype(
                            circuit_index,
                            levels,
                            shots,
                            indices,
                        )

                    if value is None:
                        value = self._execute_quantum(
                            self.circuits[circuit_index],
                            patch,
                            shots,
                        )
                        self.counters["quantum_executions"] += 1
                        self._record_shot_count(shots)
                        self.look_up[patch_key] = value
                        self._add_similarity_prototype(
                            circuit_index,
                            levels,
                            shots,
                            indices,
                            value,
                        )
                    else:
                        self.counters["similarity_cache_hits"] += 1

                    output[i, 0, h, w] = value

        return output

    # =============================================================
    # GENERATE FILTER / FILTER SELECTION
    # =============================================================

    def generate_filter(self, seed=None):
        if self.encoding_approach == constants.CircuitEncoding.INTEGRATED:
            return circuit_builder.generate_circuit_integrated(
                self.encoding_config,
                self.kernel_size,
                feature_weights=self.feature_importance,
                seed=seed,
            )

        if self.encoding_approach == constants.CircuitEncoding.ROTATIONAL:
            return circuit_builder.generate_circuit_rotational(
                self.encoding_config
            )

        if self.encoding_approach == constants.CircuitEncoding.PASS:
            return 0

        raise Exception(
            f"Encoding approach '{self.encoding_approach}' not recognized."
        )

    @staticmethod
    def _safe_abs_corr(a, b):
        a = np.asarray(a, dtype=np.float64)
        b = np.asarray(b, dtype=np.float64)
        if np.std(a) < 1e-12 or np.std(b) < 1e-12:
            return 0.0
        corr = np.corrcoef(a, b)[0, 1]
        if not np.isfinite(corr):
            return 0.0
        return abs(float(corr))

    @staticmethod
    def _candidate_quality(outputs, labels):
        """
        Score one scalar quantum filter by class separability.

        Fisher-style between-class / within-class separation is a more direct
        proxy for downstream classification utility than raw output variance.
        A small variance term prevents nearly constant filters from winning
        when the calibration set is tiny.
        """
        outputs = np.asarray(outputs, dtype=np.float64).reshape(-1)
        labels = np.asarray(labels).reshape(-1)

        if len(outputs) == 0 or len(labels) != len(outputs):
            return 0.0

        global_mean = float(outputs.mean())
        between = 0.0
        within = 0.0

        unique_labels = np.unique(labels)
        for label in unique_labels:
            values = outputs[labels == label]
            if values.size == 0:
                continue
            weight = float(values.size) / float(outputs.size)
            class_mean = float(values.mean())
            between += weight * (class_mean - global_mean) ** 2
            within += weight * float(np.var(values))

        fisher = between / (within + 1e-6)
        variance_bonus = 0.05 * float(np.var(outputs))
        return float(fisher + variance_bonus)

    def _select_diverse_filters(self, candidates):
        if not candidates:
            return []

        patches = self.filter_calibration_patches
        labels = self.filter_calibration_labels
        if len(patches) == 0:
            selected = candidates[: self.out_channels]
            self.filter_selection_summary = {
                "enabled": False,
                "reason": "No calibration patches available; used first candidates.",
            }
            return selected

        candidate_outputs = []
        for candidate_index, circuit in enumerate(candidates):
            outputs = []
            for patch in patches:
                prepared, _levels, _indices, _complexity, _shots = (
                    self._prepare_patch(patch)
                )
                value = self._execute_quantum(
                    circuit,
                    prepared,
                    self.filter_screening_shots,
                )
                self.counters["screening_executions"] += 1
                outputs.append(value)

            candidate_outputs.append(np.asarray(outputs, dtype=np.float64))

        quality = np.asarray(
            [
                self._candidate_quality(outputs, labels)
                for outputs in candidate_outputs
            ],
            dtype=np.float64,
        )

        quality_range = float(quality.max() - quality.min())
        if quality_range > 1e-12:
            quality_norm = (quality - quality.min()) / quality_range
        else:
            quality_norm = np.ones_like(quality)

        selected_indices = []
        remaining = set(range(len(candidates)))
        # Deterministic first choice: highest quality, then lowest index.
        first = min(
            remaining,
            key=lambda index: (-float(quality_norm[index]), int(index)),
        )
        selected_indices.append(first)
        remaining.remove(first)

        while len(selected_indices) < self.out_channels and remaining:
            best_index = None
            best_score = -float("inf")

            for candidate_index in remaining:
                max_corr = max(
                    self._safe_abs_corr(
                        candidate_outputs[candidate_index],
                        candidate_outputs[selected],
                    )
                    for selected in selected_indices
                )
                diversity = 1.0 - max_corr
                score = (
                    self.filter_quality_weight * quality_norm[candidate_index]
                    + self.filter_diversity_weight * diversity
                )

                if score > best_score:
                    best_score = score
                    best_index = candidate_index

            if best_index is None:
                break

            selected_indices.append(best_index)
            remaining.remove(best_index)

        selected_quality = [float(quality[index]) for index in selected_indices]
        self.filter_selection_summary = {
            "enabled": True,
            "candidate_count": len(candidates),
            "selected_indices": selected_indices,
            "selected_quality_scores": selected_quality,
            "quality_scores": quality.tolist(),
            "quality_weight": self.filter_quality_weight,
            "diversity_weight": self.filter_diversity_weight,
            "screening_shots": self.filter_screening_shots,
            "calibration_patches": len(patches),
        }

        return [candidates[index] for index in selected_indices]

    def generate_layer(self):
        if (
            self.encoding_approach == constants.CircuitEncoding.INTEGRATED
            and self.filter_selection_enabled
            and self.out_channels > 1
        ):
            candidates = [
                self.generate_filter(seed=self.quantum_seed + candidate_index)
                for candidate_index in range(self.filter_candidates)
            ]
            selected = self._select_diverse_filters(candidates)
            if len(selected) != self.out_channels:
                raise RuntimeError(
                    "Filter selection returned an incorrect number of quantum "
                    f"filters: expected {self.out_channels}, got {len(selected)}."
                )
            return selected

        layer = []
        for channel in range(self.out_channels):
            seed = self.quantum_seed + channel
            layer.append(self.generate_filter(seed=seed))

        self.filter_selection_summary = {
            "enabled": False,
            "reason": "Filter selection disabled or not applicable.",
        }
        return layer

    # =============================================================
    # QUANTUM EXECUTION / DECODING
    # =============================================================

    def _record_shot_count(self, shots):
        key = f"shots_{int(shots)}"
        if key in self.counters:
            self.counters[key] += 1

    def _execute_quantum(self, circuit, patch, shots):
        bound_circuit = circuit.assign_parameters(
            {
                f"p{i}": float(patch.reshape(-1)[i])
                for i in range(self.kernel_size ** 2)
            }
        )

        bound_circuit.measure_all()

        result = self.sampler.run(
            [bound_circuit],
            shots=int(shots),
        ).result()

        counts = result[0].data.meas.get_counts()
        if not counts:
            raise RuntimeError("Quantum sampler returned no measurement counts.")

        num_qubits = int(bound_circuit.num_qubits)
        total_shots = sum(int(count) for count in counts.values())
        if total_shots <= 0:
            raise RuntimeError("Quantum sampler returned zero total shots.")

        sum_ones = sum(
            key.count("1") * int(count)
            for key, count in counts.items()
        )

        # Correct normalization: average fraction of measured 1-bits over
        # the actual number of qubits, not kernel_size^2.
        return float(sum_ones / total_shots / num_qubits)

    def to_quanvolute_patch(self, circuit, patch):
        """Compatibility wrapper preserving the original public method."""
        if self.encoding_approach not in (
            constants.CircuitEncoding.INTEGRATED,
            constants.CircuitEncoding.ROTATIONAL,
        ):
            if self.encoding_approach == constants.CircuitEncoding.PASS:
                return 0
            raise Exception(
                f"Encoding approach '{self.encoding_approach}' not recognized."
            )

        if self.encoding_config is None:
            self.encoding_config = utils.default_encoding_config(
                self.encoding_approach
            )

        shots = self.encoding_config.get("n_shots", 1)
        return self._execute_quantum(
            circuit,
            np.asarray(patch, dtype=np.float32),
            int(shots),
        )

    # =============================================================
    # PRINT COUNTERS
    # =============================================================

    def print_counters(self):
        print(self.counters)
