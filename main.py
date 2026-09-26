# main.py
import json
import os
import random
from typing import Any, cast

import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader

from quanvs.model_builder import CNN
from quanvs.Quanvolutional_Layer import QuanvolutionalLayer
from quanvs.model_builder import stack_quanv_on_top

from utils.read_config import load_config
from utils.get_dataset.dataset_Mirabest import get_MiraBest_binary
from utils.get_dataset.dataset_LArTPC import get_LArTPC_full

from utils.train_and_test import train, test
from quanvs.quanv_util import analyze_training_patches


# ============================================================
# EXPERIMENT SETTINGS
# ============================================================

DEBUG_MODE = False
SELECTED_TASK = "MiraBest"

# Run the complete fair comparison in one invocation.
# CNN is the unchanged classical baseline; the two QNN variants share the
# same data-aware quantum preprocessing procedure.
CONFIGURATIONS_TO_RUN = (
    "CNN",
    "QNN-Int-Simple-k3",
    "QNN-Int-RndMul-k3",
)

# Ten independent classical seeds.
SEEDS = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

# Fallback only when adaptive quantization is disabled.
QUANTIZATION = 50

# Quanvolutional output channels.
N_QUANV_CHANNELS = 8

# Final-comparison training budget.
# Keep the same budget for CNN, Simple and RndMul.
EPOCHS = 1000

LEARNING_RATE = 0.0003

# One fixed quantum seed shared by all classical seeds.
QUANTUM_SEED = 2026


# ============================================================
# RESEARCH METHOD SETTINGS
# ============================================================

RESEARCH_CONFIG = {
    # Report-aligned resource-aware run:
    # all five proposed research mechanisms are ACTIVE.
    # Adaptive quantization, similarity-aware memoization and adaptive shots
    # are enabled for the same fixed quantum circuits used by each model.
    "adaptive_quantization": True,
    "quantization_levels": (10, 25, 50),

    # Similarity-aware memoization is enabled after exact-cache misses.
    # Candidate reuse is validated for a bounded number of cache events before
    # being accepted, keeping approximation under control.
    "similarity_memoization": True,
    "similarity_threshold": 0.05,
    "similarity_coarse_levels": 8,
    "max_prototypes_per_bucket": 32,
    "similarity_validation_limit": 64,
    "similarity_output_tolerance": 0.05,

    # 3) Diversity-aware selection of the eight quantum filters.
    # More candidates + stronger screening gives the selector a better chance
    # of finding useful and non-redundant filters without changing the final
    # architecture (still exactly 8 output channels).
    "filter_selection": True,
    "filter_candidates": 32,
    "filter_screening_shots": 250,
    "filter_quality_weight": 0.65,
    "filter_diversity_weight": 0.35,

    # 4) Global training-derived information-aware gate allocation.
    # Every feature still appears at least once and L stays fixed by YAML.
    "information_aware_gates": True,

    # Adaptive measurement: lower-precision patches use fewer shots while
    # high-precision patches retain the full measurement budget.
    "adaptive_shots": True,
    "shots_by_level": {
        10: 250,
        25: 500,
        50: 1000,
    },

    "quantum_seed": QUANTUM_SEED,
}


# ============================================================
# OUTPUT DIRECTORY
# ============================================================

TASK_PATH = f"exps/{SELECTED_TASK}"
os.makedirs(TASK_PATH, exist_ok=True)


# ============================================================
# REPRODUCIBILITY
# ============================================================

def set_classical_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def make_seeded_loader(dataset, batch_size, seed, shuffle):
    generator = torch.Generator()
    generator.manual_seed(int(seed))

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        generator=generator,
        pin_memory=torch.cuda.is_available(),
    )


# ============================================================
# CLASSICAL TRAINING
# ============================================================

def run_classical_training(
    model,
    device,
    train_dataset,
    test_dataset,
    batch_size,
    seed,
    optimizer,
    seed_path,
):
    """
    Final-comparison training:
      - full original training set
      - no validation split
      - no early stopping
      - same epoch budget for every classical seed
      - diagnostic test evaluation every 25 epochs and at the end
    """
    set_classical_seed(seed)

    train_loader = make_seeded_loader(
        train_dataset,
        batch_size,
        seed,
        shuffle=True,
    )

    test_loader = make_seeded_loader(
        test_dataset,
        batch_size,
        seed,
        shuffle=False,
    )

    data_lines = [
        "epoch,train_loss,train_acc,test_loss,test_acc\n"
    ]

    for epoch in range(1, EPOCHS + 1):
        train_loss, train_acc = train(
            model,
            device,
            train_loader,
            optimizer,
            epoch,
            verbose=False,
        )

        # Keep the test set out of the inner loop for most epochs.
        # For the final comparison, inspect it only every 25 epochs and at
        # the final epoch. The selected report metric remains final epoch.
        if epoch == 1 or epoch % 25 == 0 or epoch == EPOCHS:
            test_loss, test_acc = test(
                model,
                device,
                test_loader,
                verbose=False,
            )

            data_lines.append(
                f"{epoch},{train_loss},{train_acc},"
                f"{test_loss},{test_acc}\n"
            )

            print(
                f"Seed {seed} | "
                f"Epoch {epoch}/{EPOCHS} | "
                f"Train loss: {train_loss:.4f} | "
                f"Train acc: {train_acc:.2f}% | "
                f"Test acc: {test_acc:.2f}%"
            )
        else:
            # Record train-only progress without touching the test set.
            data_lines.append(
                f"{epoch},{train_loss},{train_acc},,\n"
            )

    final_test_loss, final_test_acc = test(
        model,
        device,
        test_loader,
        verbose=True,
    )

    with open(
        os.path.join(seed_path, "losses.txt"),
        "w",
        encoding="utf-8",
    ) as f:
        f.writelines(data_lines)

    with open(
        os.path.join(seed_path, "selected_metrics.txt"),
        "w",
        encoding="utf-8",
    ) as f:
        f.write("selection_protocol=final_epoch\n")
        f.write(f"final_epoch={EPOCHS}\n")
        f.write(f"final_test_loss={final_test_loss}\n")
        f.write(f"final_test_accuracy={final_test_acc}\n")

    return final_test_loss, final_test_acc


# ============================================================
# DATASET
# ============================================================

def load_dataset():
    if SELECTED_TASK == "LArTPC":
        return get_LArTPC_full(
            downscale=True,
            autocrop=False,
        )

    if SELECTED_TASK == "MiraBest":
        return get_MiraBest_binary()

    raise ValueError(
        f"Unknown task: {SELECTED_TASK}. "
        f"Use 'MiraBest' or 'LArTPC'."
    )


# ============================================================
# REPORT / CODE COMPATIBILITY CHECKS
# ============================================================

def validate_report_alignment(encoding, quanv_config, kernel_size):
    """Validate and normalize the report-defined integrated-encoding contract."""
    encoding_name = getattr(encoding, "value", encoding)
    if str(encoding_name).lower() != "integrated":
        return

    expected_l = 2 * int(kernel_size) ** 2
    configured_l = int(quanv_config.get("L", expected_l))

    # The report defines L=2*k^2 for integrated encoding. Normalize the runtime
    # configuration here so an older YAML value cannot silently invalidate the
    # reported method. The final configuration is then validated below.
    if configured_l != expected_l:
        print(
            f"WARNING: overriding configured L={configured_l} with "
            f"report-aligned L={expected_l} for k={kernel_size}."
        )
        quanv_config["L"] = expected_l

    if int(quanv_config["L"]) != expected_l:
        raise ValueError(
            "Integrated encoding gate-budget validation failed: "
            f"expected L={expected_l}, got {quanv_config['L']}."
        )

    levels = tuple(sorted(int(x) for x in RESEARCH_CONFIG["quantization_levels"]))
    if levels != (10, 25, 50):
        raise ValueError(
            "Report/code mismatch: adaptive quantization must use N={10,25,50}."
        )

    required_shots = {10: 250, 25: 500, 50: 1000}
    actual_shots = {
        int(k): int(v)
        for k, v in RESEARCH_CONFIG["shots_by_level"].items()
    }
    if actual_shots != required_shots:
        raise ValueError(
            "Report/code mismatch: adaptive shots must be {10:250, 25:500, 50:1000}."
        )

    if not RESEARCH_CONFIG["adaptive_quantization"]:
        raise ValueError("Adaptive quantization is disabled in the active configuration.")
    if not RESEARCH_CONFIG["similarity_memoization"]:
        raise ValueError("Similarity-aware memoization is disabled in the active configuration.")
    if not RESEARCH_CONFIG["filter_selection"]:
        raise ValueError("Diversity-aware filter selection is disabled in the active configuration.")
    if not RESEARCH_CONFIG["information_aware_gates"]:
        raise ValueError("Information-aware gate allocation is disabled in the active configuration.")
    if not RESEARCH_CONFIG["adaptive_shots"]:
        raise ValueError("Adaptive measurement shots are disabled in the active configuration.")


# ============================================================
# MAIN
# ============================================================

def main():
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print("\n" + "=" * 70)
    print(f"TASK: {SELECTED_TASK}")
    print(f"DEVICE: {device}")

    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    print(f"CONFIGURATIONS: {CONFIGURATIONS_TO_RUN}")
    print(f"CLASSICAL SEEDS: {SEEDS}")
    print(f"FIXED QUANTUM SEED: {QUANTUM_SEED}")
    print(f"EPOCHS: {EPOCHS}")
    print("RESOURCE-AWARE METHODS: all five ENABLED")
    print("  - Adaptive quantization: N={10,25,50}")
    print("  - Similarity-aware memoization: enabled + validation")
    print("  - Diversity-aware filter selection: enabled")
    print("  - Information-aware gate allocation: enabled")
    print("  - Adaptive measurement shots: 250/500/1000")
    print("=" * 70)

    train_loader, test_loader, _ = load_dataset()

    if train_loader is None or test_loader is None:
        raise RuntimeError(
            "Training or test loader could not be created."
        )

    train_dataset = cast(Any, train_loader.dataset)
    test_dataset = cast(Any, test_loader.dataset)

    print(
        f"Training samples: {len(train_dataset)} | "
        f"Test samples: {len(test_dataset)}"
    )

    config_folder = "configs"

    if not os.path.isdir(config_folder):
        raise FileNotFoundError(
            f"Configuration folder '{config_folder}' not found."
        )

    available_files = {
        os.path.splitext(file)[0]: file
        for file in os.listdir(config_folder)
        if file.endswith((".yaml", ".yml"))
    }

    missing = [
        name
        for name in CONFIGURATIONS_TO_RUN
        if name not in available_files
    ]

    if missing:
        raise FileNotFoundError(
            "Required configuration file(s) not found: "
            + ", ".join(missing)
        )

    analysis = None

    for config_name in CONFIGURATIONS_TO_RUN:
        file = available_files[config_name]
        file_path = os.path.join(config_folder, file)

        print("\n" + "-" * 70)
        print(f"Working on {config_name} for {SELECTED_TASK}")
        print("-" * 70)

        encoding, quanv_config, _, model_config = load_config(
            file_path
        )

        model_config["fc2"]["out_features"] = (
            2 if SELECTED_TASK == "MiraBest" else 7
        )

        if quanv_config is not None and encoding == "integrated":
            kernel_size = int(quanv_config["kernel_size"])
            validate_report_alignment(
                encoding,
                quanv_config,
                kernel_size,
            )

        config_path = os.path.join(
            TASK_PATH,
            config_name,
        )
        os.makedirs(config_path, exist_ok=True)

        # ====================================================
        # CNN BASELINE
        # ====================================================

        if quanv_config is None:
            for seed in SEEDS:
                print(f"Starting CNN seed={seed}")

                set_classical_seed(seed)

                model = CNN(
                    device=device,
                    config=model_config,
                ).to(device)

                optimizer = optim.Adam(
                    model.parameters(),
                    lr=LEARNING_RATE,
                )

                seed_path = os.path.join(
                    config_path,
                    f"seed_{seed}",
                )
                os.makedirs(seed_path, exist_ok=True)

                if DEBUG_MODE:
                    print(
                        f"DEBUG_MODE=True -> skipping CNN "
                        f"training for seed={seed}"
                    )
                else:
                    _, final_test_acc = run_classical_training(
                        model=model,
                        device=device,
                        train_dataset=train_dataset,
                        test_dataset=test_dataset,
                        batch_size=train_loader.batch_size,
                        seed=seed,
                        optimizer=optimizer,
                        seed_path=seed_path,
                    )

                    print(
                        f"CNN seed={seed}: "
                        f"final_test_accuracy="
                        f"{final_test_acc:.4f}%"
                    )

                with open(
                    os.path.join(
                        seed_path,
                        "experiment_settings.txt",
                    ),
                    "w",
                    encoding="utf-8",
                ) as f:
                    f.write(f"Task: {SELECTED_TASK}\n")
                    f.write(f"Model: {config_name}\n")
                    f.write(f"Seed: {seed}\n")
                    f.write(f"Device: {device}\n")
                    f.write(f"Epochs: {EPOCHS}\n")
                    f.write(
                        f"Learning rate: {LEARNING_RATE}\n"
                    )
                    f.write(
                        "Training protocol: full original "
                        "training split; no validation split; "
                        "no early stopping.\n"
                    )
                    f.write(
                        "Test metrics are reported at selected "
                        "diagnostic epochs and final epoch.\n"
                    )

            continue

        # ====================================================
        # TRAINING-ONLY DATA-AWARE CALIBRATION
        # ====================================================

        if analysis is None:
            kernel_size = int(
                quanv_config["kernel_size"]
            )
            padding = int((kernel_size - 1) / 2)

            analysis_loader = DataLoader(
                train_dataset,
                batch_size=train_loader.batch_size,
                shuffle=False,
            )

            print(
                f"\nAnalyzing training patches for k={kernel_size} "
                "(training data only)..."
            )

            analysis = analyze_training_patches(
                analysis_loader,
                kernel_size=kernel_size,
                padding=padding,
                stride=1,
                max_calibration_patches=64,
            )

            print(
                "Feature importance:",
                analysis["feature_importance"].tolist(),
            )

            print(
                "Adaptive quantization thresholds:",
                analysis["quantization_thresholds"],
            )

        # ====================================================
        # REPORT-ALIGNED QUANTUM FEATURE EXTRACTOR
        # ====================================================

        kernel_size = int(quanv_config["kernel_size"])
        padding = int((kernel_size - 1) / 2)

        layer = QuanvolutionalLayer(
            in_channels=1,
            out_channels=N_QUANV_CHANNELS,
            kernel_size=kernel_size,
            stride=1,
            padding=padding,
            # Used only as a fallback when adaptive quantization is disabled.
            quantization=QUANTIZATION,
            encoding_approach=encoding,
            encoding_config=quanv_config,
            research_config=RESEARCH_CONFIG,
            feature_importance=analysis["feature_importance"],
            quantization_thresholds=analysis[
                "quantization_thresholds"
            ],
            filter_calibration_patches=analysis[
                "calibration_patches"
            ],
            filter_calibration_labels=analysis[
                "calibration_labels"
            ],
        )

        preprocessing_cnn = CNN(
            device=device,
            config=model_config,
        )

        preprocessing_model = stack_quanv_on_top(
            layer,
            preprocessing_cnn,
        )

        if DEBUG_MODE:
            print(
                "DEBUG_MODE=True -> skipping QNN preprocessing "
                "and training."
            )
            continue

        print("\nRunning ONE data-aware quantum preprocessing pass...")

        preprocessed_train_loader = (
            preprocessing_model.quanv_preprocess(
                DataLoader(
                    train_dataset,
                    batch_size=train_loader.batch_size,
                    shuffle=False,
                ),
                verbose=False,
            )
        )

        preprocessed_test_loader = (
            preprocessing_model.quanv_preprocess(
                DataLoader(
                    test_dataset,
                    batch_size=test_loader.batch_size,
                    shuffle=False,
                ),
                verbose=False,
            )
        )

        preprocessed_train_dataset = (
            preprocessed_train_loader.dataset
        )
        preprocessed_test_dataset = (
            preprocessed_test_loader.dataset
        )

        preprocessing_model.preprocessed = True

        with open(
            os.path.join(
                config_path,
                "research_method.json",
            ),
            "w",
            encoding="utf-8",
        ) as f:
            json.dump(
                {
                    "task": SELECTED_TASK,
                    "model": config_name,
                    "quantum_seed": QUANTUM_SEED,
                    "classical_seeds": SEEDS,
                    "epochs": EPOCHS,
                    "research_config": RESEARCH_CONFIG,
                    "feature_importance": analysis[
                        "feature_importance"
                    ].tolist(),
                    "quantization_thresholds": analysis[
                        "quantization_thresholds"
                    ],
                    "num_training_patches_analyzed": analysis[
                        "num_training_patches"
                    ],
                    "filter_selection_summary": (
                        layer.filter_selection_summary
                    ),
                    "counters": layer.counters,
                },
                f,
                indent=2,
            )

        with open(
            os.path.join(
                config_path,
                "structure.txt",
            ),
            "w",
            encoding="utf-8",
        ) as f:
            f.write(str(preprocessing_model))

        with open(
            os.path.join(
                config_path,
                "quanv_structure.txt",
            ),
            "w",
            encoding="utf-8",
        ) as f:
            f.write(str(layer))

        with open(
            os.path.join(
                config_path,
                "quanv_encoding_config.txt",
            ),
            "w",
            encoding="utf-8",
        ) as f:
            f.write(str(layer.encoding_config))
            f.write("\n")
            f.write(str(layer.counters))
            f.write("\n")
            f.write(str(quanv_config))
            f.write("\n")
            f.write(str(layer.filter_selection_summary))

        # ====================================================
        # TEN CLASSICAL RUNS ON SAME FIXED QUANTUM FEATURES
        # ====================================================

        for seed in SEEDS:
            print(
                f"Starting {config_name}, seed={seed} "
                "(quantum preprocessing reused)"
            )

            set_classical_seed(seed)

            model = CNN(
                device=device,
                config=model_config,
            )

            model = stack_quanv_on_top(
                layer,
                model,
            )

            model.preprocessed = True
            model.to(device)

            optimizer = optim.Adam(
                model.parameters(),
                lr=LEARNING_RATE,
            )

            seed_path = os.path.join(
                config_path,
                f"seed_{seed}",
            )
            os.makedirs(seed_path, exist_ok=True)

            with open(
                os.path.join(
                    seed_path,
                    "experiment_settings.txt",
                ),
                "w",
                encoding="utf-8",
            ) as f:
                f.write(f"Task: {SELECTED_TASK}\n")
                f.write(f"Model: {config_name}\n")
                f.write(f"Seed: {seed}\n")
                f.write(f"Device: {device}\n")
                f.write(
                    "Quantum preprocessing: "
                    "fixed/shared across seeds\n"
                )
                f.write(f"Quantum seed: {QUANTUM_SEED}\n")
                f.write(
                    "Training protocol: full original "
                    "training split; no validation split; "
                    "no early stopping.\n"
                )
                f.write(f"Epochs: {EPOCHS}\n")
                f.write(
                    f"Learning rate: {LEARNING_RATE}\n"
                )

                f.write("\nQuantum configuration:\n")
                for key, value in quanv_config.items():
                    f.write(f"{key}: {value}\n")

                f.write("\nResearch configuration:\n")
                for key, value in RESEARCH_CONFIG.items():
                    f.write(f"{key}: {value}\n")

            if DEBUG_MODE:
                print(
                    f"DEBUG_MODE=True -> skipping seed={seed}"
                )
            else:
                _, final_test_acc = run_classical_training(
                    model=model,
                    device=device,
                    train_dataset=preprocessed_train_dataset,
                    test_dataset=preprocessed_test_dataset,
                    batch_size=train_loader.batch_size,
                    seed=seed,
                    optimizer=optimizer,
                    seed_path=seed_path,
                )

                print(
                    f"{config_name} seed={seed}: "
                    f"final_test_accuracy="
                    f"{final_test_acc:.4f}%"
                )

            print(
                f"Completed {config_name}, seed={seed} "
                "(quantum preprocessing reused)"
            )

    print("\n" + "=" * 70)
    print("CNN + QNN-Int-Simple-k3 + QNN-Int-RndMul-k3 COMPLETED")
    print("=" * 70)


if __name__ == "__main__":
    main()
