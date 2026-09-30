import os
import pandas as pd


# ============================================================
# SETTINGS
# ============================================================

SELECTED_TASK = "MNIST"
EXPS_DIR = f"exps/{SELECTED_TASK}"

NUM_SEEDS = 10

# Primary reporting protocol:
# use the explicitly saved final-test result from each seed.
USE_FINAL_EPOCH = True


# ============================================================
# HELPERS
# ============================================================

def read_final_accuracy(seed_folder):
    """
    Read the explicitly saved final test accuracy.

    Expected file:
        selected_metrics.txt

    Expected line:
        final_test_accuracy=<value>
    """

    metrics_file = os.path.join(
        seed_folder,
        "selected_metrics.txt",
    )

    if not os.path.exists(metrics_file):
        return None

    try:
        with open(
            metrics_file,
            "r",
            encoding="utf-8",
        ) as f:
            for line in f:
                line = line.strip()

                if line.startswith("final_test_accuracy="):
                    value = line.split(
                        "=",
                        1,
                    )[1].strip()

                    return float(value)

    except (OSError, ValueError):
        return None

    return None


def read_last_valid_test_accuracy(seed_folder):
    """
    Fallback for older experiments that do not have
    selected_metrics.txt.

    It reads losses.txt and uses the LAST valid test_acc
    entry rather than accidentally selecting a NaN row.
    """

    loss_file = os.path.join(
        seed_folder,
        "losses.txt",
    )

    if not os.path.exists(loss_file):
        return None

    try:
        df = pd.read_csv(loss_file)

        if "test_acc" not in df.columns:
            return None

        valid_accuracy = pd.to_numeric(
            df["test_acc"],
            errors="coerce",
        ).dropna()

        if valid_accuracy.empty:
            return None

        return float(valid_accuracy.iloc[-1])

    except (
        OSError,
        ValueError,
        pd.errors.EmptyDataError,
        pd.errors.ParserError,
    ):
        return None


def read_seed_accuracy(seed_folder):
    """
    Read the correct final accuracy for one seed.

    Priority:
        1. selected_metrics.txt
        2. last valid test_acc in losses.txt
    """

    final_accuracy = read_final_accuracy(seed_folder)

    if final_accuracy is not None:
        return final_accuracy

    return read_last_valid_test_accuracy(seed_folder)


def population_std(values):
    """
    Standard deviation using population definition.

    This preserves the behavior of your original script.
    """

    if not values:
        return 0.0

    mean_value = sum(values) / len(values)

    variance = sum(
        (value - mean_value) ** 2
        for value in values
    ) / len(values)

    return variance ** 0.5


# ============================================================
# CHECK EXPERIMENT DIRECTORY
# ============================================================

if not os.path.isdir(EXPS_DIR):
    raise FileNotFoundError(
        f"Experiment directory not found: {EXPS_DIR}"
    )


# ============================================================
# COLLECT RESULTS
# ============================================================

results = []

folders = sorted(
    os.listdir(EXPS_DIR)
)


for folder in folders:

    model_folder = os.path.join(
        EXPS_DIR,
        folder,
    )

    if not os.path.isdir(model_folder):
        continue

    accuracies = []

    print("\n" + "=" * 70)
    print(f"Model: {folder}")
    print("=" * 70)

    for seed in range(NUM_SEEDS):

        seed_folder = os.path.join(
            model_folder,
            f"seed_{seed}",
        )

        if not os.path.isdir(seed_folder):
            print(
                f"Seed {seed}: folder not found"
            )
            continue

        accuracy = read_seed_accuracy(
            seed_folder
        )

        if accuracy is None:
            print(
                f"Seed {seed}: accuracy not found"
            )
            continue

        accuracies.append(accuracy)

        print(
            f"Seed {seed}: "
            f"{accuracy:.4f}%"
        )

    # --------------------------------------------------------
    # Aggregate
    # --------------------------------------------------------

    n_seeds = len(accuracies)

    if n_seeds == 0:
        print(
            "No valid seed results found. Skipping."
        )
        continue

    average_accuracy = (
        sum(accuracies) / n_seeds
    )

    std_accuracy = population_std(
        accuracies
    )

    print(
        f"\nMean: {average_accuracy:.4f}%"
    )

    print(
        f"Std:  {std_accuracy:.4f}%"
    )

    print(
        f"Seeds used: {n_seeds}"
    )

    results.append(
        (
            folder,
            n_seeds,
            average_accuracy,
            std_accuracy,
        )
    )


# ============================================================
# SORT RESULTS
# ============================================================

results.sort(
    key=lambda row: row[0]
)


# ============================================================
# PRINT SUMMARY
# ============================================================

print("\n")
print("=" * 98)
print("FINAL RESULTS")
print("=" * 98)

print(
    f"{'Model_Encoding':<35}"
    f"{'Num_Seeds':<12}"
    f"{'Average_Accuracy':<22}"
    f"{'Std_Accuracy':<20}"
)

print("-" * 98)

for (
    model_encoding,
    num_seeds,
    average_accuracy,
    std_accuracy,
) in results:

    print(
        f"{model_encoding:<35}"
        f"{num_seeds:<12}"
        f"{average_accuracy:<22.4f}"
        f"{std_accuracy:<20.4f}"
    )


# ============================================================
# SAVE SIMPLE SUMMARY
# ============================================================

summary_df = pd.DataFrame(
    results,
    columns=[
        "Model_Encoding",
        "Num_Seeds",
        "Average_Accuracy",
        "Std_Accuracy",
    ],
)

summary_df.to_csv(
    "sorted_best_test_accuracies.csv",
    index=False,
)


# ============================================================
# BUILD RESULTS TABLE
# ============================================================

def extract_k_value(model_name):
    """
    Extract kernel size where possible.

    Examples:
        integrated-k3-Fixed
        integrated-k3-Full
        QNN-Int-Simple-k3
        QNN-Int-RndMul-k3

    For CNN, use 0.
    """

    parts = model_name.split("-")

    for part in parts:

        if part.startswith("k"):

            numeric_part = part[1:]

            try:
                return int(numeric_part)

            except ValueError:
                pass

    return 0


summary_df["k_value"] = (
    summary_df["Model_Encoding"]
    .apply(extract_k_value)
)


summary_df["model_name"] = (
    summary_df["Model_Encoding"]
)


# ============================================================
# PIVOT ACCURACY
# ============================================================

accuracy_table = summary_df.pivot(
    index="model_name",
    columns="k_value",
    values="Average_Accuracy",
)


# ============================================================
# PIVOT STANDARD DEVIATION
# ============================================================

std_table = summary_df.pivot(
    index="model_name",
    columns="k_value",
    values="Std_Accuracy",
)


std_table.columns = [
    f"{column}_std"
    for column in std_table.columns
]


data = pd.concat(
    [
        accuracy_table,
        std_table,
    ],
    axis=1,
)


# ============================================================
# SAVE FINAL RESULTS
# ============================================================

RESULTS_DIR = "results"

os.makedirs(
    RESULTS_DIR,
    exist_ok=True,
)


output_file = os.path.join(
    RESULTS_DIR,
    f"{SELECTED_TASK}.csv",
)


data.to_csv(
    output_file,
    index=True,
)


# ============================================================
# FINAL MESSAGE
# ============================================================

print(
    f"\nSaved summary to: "
    f"{output_file}"
)