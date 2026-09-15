#circuit_builder.py
from qiskit.circuit import QuantumCircuit, Parameter

import random
import math
import numpy as np
import constants



# =============================================================
# RESEARCH EXTENSIONS: WEIGHTED FEATURE ALLOCATION
# =============================================================

def _weighted_choice(rng, weights):
    """
    Select an index according to non-negative weights.

    This is used only when information-aware feature weights are supplied.
    If no valid positive weights are available, selection falls back to
    uniform random selection.
    """
    if not weights:
        raise ValueError("weights cannot be empty.")

    clean_weights = [max(float(weight), 0.0) for weight in weights]
    total = sum(clean_weights)

    if total <= 0.0:
        return rng.randrange(len(clean_weights))

    target = rng.random() * total
    cumulative = 0.0

    for index, weight in enumerate(clean_weights):
        cumulative += weight
        if target <= cumulative:
            return index

    return len(clean_weights) - 1


# =============================================================
# INTEGRATED ENCODING
# =============================================================



def _prepare_feature_weights(feature_weights, feature_count):
    """Return stable, non-degenerate weights for the fixed gate allocator."""
    if feature_weights is None:
        return [1.0] * feature_count, False

    if len(feature_weights) != feature_count:
        raise ValueError("feature_weights length must equal kernel_size^2.")

    values = np.asarray(feature_weights, dtype=np.float64)
    values = np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)
    values = np.maximum(values, 0.0)

    if float(values.sum()) <= 0.0:
        return [1.0] * feature_count, False

    # Small floor keeps every feature eligible while still giving the most
    # informative positions extra probability for the remaining gates.
    values = values + 0.10 * float(values.mean())
    values = values / float(values.sum())
    return values.tolist(), True


def generate_circuit_integrated(
    encoding_config,
    kernel_size,
    feature_weights=None,
    seed=None
):
    """
    Generate the integrated encoding quantum circuit.

    Parameters
    ----------
    encoding_config : dict
        Quantum circuit configuration.

    kernel_size : int
        Quanvolutional kernel size.

    feature_weights : list or None
        Optional training-derived feature-importance weights. The first
        k^2 gates still guarantee one occurrence of every feature; remaining
        gates are allocated according to these weights.

    seed : int or None
        Optional local random seed for reproducible fixed circuit generation.

    Returns
    -------
    QuantumCircuit
        Parameterized integrated quantum circuit.
    """

    n_qubits = int(encoding_config["n_qubits"])
    L = int(encoding_config["L"])
    activation = encoding_config["activation"]
    feature_count = int(kernel_size ** 2)

    if L < feature_count:
        raise ValueError(
            f"Integrated circuit requires L >= kernel_size^2 "
            f"({feature_count}), but received L={L}."
        )

    # A local RNG keeps circuit generation reproducible without changing
    # the global random state used elsewhere in the experiment.
    rng = random.Random(seed) if seed is not None else random

    weights, use_weighted_allocation = _prepare_feature_weights(
        feature_weights,
        feature_count,
    )

    # ---------------------------------------------------------
    # Quantum circuit
    # ---------------------------------------------------------

    circuit = QuantumCircuit(
        n_qubits
    )

    # One data parameter per input pixel
    data = [
        Parameter(f"p{i}")
        for i in range(feature_count)
    ]

    # Preserve original parameter structure
    for par in data:

        circuit.rx(
            par * 0,
            0
        )

    # ---------------------------------------------------------
    # Generate random gates
    # ---------------------------------------------------------

    gates = []

    # Guarantee that every input feature is represented at least once.
    # The mandatory phase is kept exactly so the original integrated-encoding
    # contract is never violated.
    mandatory_indices = list(range(feature_count))
    rng.shuffle(mandatory_indices)

    while len(gates) < L:

        q1 = rng.randint(
            0,
            n_qubits - 1
        )

        q2 = rng.randint(
            0,
            n_qubits - 1
        )

        # Do not use the same qubit twice
        if q1 != q2:

            # Fixed random multiplier stored with this gate.
            beta = rng.random()

            if len(gates) < feature_count:

                # Preserve the paper's "every feature at least once"
                # requirement, but randomize which mandatory feature is
                # placed at each of the first feature slots.
                index = mandatory_indices[len(gates)]

            else:

                # Research modification:
                # allocate the remaining gates according to global,
                # training-derived feature importance.
                if use_weighted_allocation:
                    index = _weighted_choice(rng, weights)
                else:
                    index = rng.randrange(feature_count)

            # -------------------------------------------------
            # Random Pauli generators
            # -------------------------------------------------

            g1 = "I"
            g2 = "I"

            while (
                g1 == "I"
                and g2 == "I"
            ):

                g1 = rng.choice(
                    ["I", "X", "Y", "Z"]
                )

                g2 = rng.choice(
                    ["I", "X", "Y", "Z"]
                )

            gates.append(
                {
                    "q1": q1,
                    "q2": q2,
                    "g1": g1,
                    "g2": g2,
                    "beta": beta,
                    "index": index
                }
            )

    # ---------------------------------------------------------
    # Shuffle generated gates
    # ---------------------------------------------------------

    rng.shuffle(gates)

    # ---------------------------------------------------------
    # Add gates to circuit
    # ---------------------------------------------------------

    for gate in gates:

        q1 = gate["q1"]
        q2 = gate["q2"]
        g1 = gate["g1"]
        g2 = gate["g2"]

        # IMPORTANT:
        # Retrieve the beta belonging to this gate
        # BEFORE calculating the rotation parameter.
        beta = gate["beta"]
        index = gate["index"]

        # -----------------------------------------------------
        # Pixel -> quantum rotation parameter
        # -----------------------------------------------------

        if (
            activation
            == constants.QuanvActivation.FULL.value
        ):

            # RndMul:
            # alpha(x) = 2 * beta * x * pi
            param = (
                data[index]
                * beta
                * 2
                * math.pi
            )

        elif (
            activation
            == constants.QuanvActivation.HALF.value
        ):

            param = (
                data[index]
                * beta
                * math.pi
            )

        elif (
            activation
            == constants.QuanvActivation.SHIFTED.value
        ):

            param = (
                data[index]
                * beta
                * math.pi
                + math.pi / 2
            )

        elif (
            activation
            == constants.QuanvActivation.RANDOM.value
        ):

            param = (
                data[index]
                * beta
                * math.pi
                + rng.random() * math.pi
            )

        elif (
            activation
            == constants.QuanvActivation.FIXED.value
        ):

            # Simple:
            # alpha(x) = x * pi
            param = (
                data[index]
                * math.pi
            )

        else:

            raise ValueError(
                f"Activation '{activation}' not supported"
            )

        # -----------------------------------------------------
        # Quantum generator combinations
        # -----------------------------------------------------

        if g1 == g2:

            # X ⊗ X
            if g1 == "X":

                circuit.rxx(
                    param,
                    q1,
                    q2
                )

            # Y ⊗ Y
            elif g1 == "Y":

                circuit.ryy(
                    param,
                    q1,
                    q2
                )

            # Z ⊗ Z
            elif g1 == "Z":

                circuit.rzz(
                    param,
                    q1,
                    q2
                )

        # -----------------------------------------------------
        # I ⊗ X/Y/Z
        # -----------------------------------------------------

        elif g1 == "I":

            if g2 == "X":

                circuit.rx(
                    param,
                    q2
                )

            elif g2 == "Y":

                circuit.ry(
                    param,
                    q2
                )

            elif g2 == "Z":

                circuit.rz(
                    param,
                    q2
                )

        # -----------------------------------------------------
        # X/Y/Z ⊗ I
        # -----------------------------------------------------

        elif g2 == "I":

            if g1 == "X":

                circuit.rx(
                    param,
                    q1
                )

            elif g1 == "Y":

                circuit.ry(
                    param,
                    q1
                )

            elif g1 == "Z":

                circuit.rz(
                    param,
                    q1
                )

        # -----------------------------------------------------
        # Z ⊗ X
        # -----------------------------------------------------

        elif (
            g1 == "Z"
            and g2 == "X"
        ):

            circuit.rzx(
                param,
                q1,
                q2
            )

        # -----------------------------------------------------
        # X ⊗ Z
        # -----------------------------------------------------

        elif (
            g1 == "X"
            and g2 == "Z"
        ):

            circuit.rzx(
                param,
                q2,
                q1
            )

        # -----------------------------------------------------
        # X ⊗ Y
        # -----------------------------------------------------

        elif (
            g1 == "X"
            and g2 == "Y"
        ):

            circuit.rzz(
                math.pi / 4,
                q1,
                q2
            )

            circuit.rxx(
                param,
                q1,
                q2
            )

            circuit.rzz(
                -math.pi / 4,
                q1,
                q2
            )

        # -----------------------------------------------------
        # Y ⊗ X
        # -----------------------------------------------------

        elif (
            g1 == "Y"
            and g2 == "X"
        ):

            circuit.rzz(
                math.pi / 4,
                q2,
                q1
            )

            circuit.rxx(
                param,
                q2,
                q1
            )

            circuit.rzz(
                -math.pi / 4,
                q2,
                q1
            )

        # -----------------------------------------------------
        # Y ⊗ Z
        # -----------------------------------------------------

        elif (
            g1 == "Y"
            and g2 == "Z"
        ):

            circuit.rzz(
                math.pi / 4,
                q2,
                q1
            )

            circuit.ryy(
                param,
                q2,
                q1
            )

            circuit.rzz(
                -math.pi / 4,
                q2,
                q1
            )

        # -----------------------------------------------------
        # Z ⊗ Y
        # -----------------------------------------------------

        elif (
            g1 == "Z"
            and g2 == "Y"
        ):

            circuit.rzz(
                math.pi / 4,
                q1,
                q2
            )

            circuit.ryy(
                param,
                q2,
                q1
            )

            circuit.rzz(
                -math.pi / 4,
                q1,
                q2
            )

    return circuit


# =============================================================
# ROTATIONAL ENCODING
# =============================================================

def generate_circuit_rotational(
    encoding_config
):

    connection_prob = (
        encoding_config["probability"]
    )

    n_qubits = (
        encoding_config["n_qubits"]
    )

    activation = (
        encoding_config["activation"]
    )

    one_qb_list = [
        "X",
        "Y",
        "Z",
        "P",
        "T",
        "H"
    ]

    two_qb_list = [
        "Cnot",
        "Swap",
        "SqrtSwap"
    ]

    gate_list = []

    # ---------------------------------------------------------
    # Generate two-qubit connections
    # ---------------------------------------------------------

    for i in range(n_qubits):

        for j in range(n_qubits):

            if (
                i != j
                and random.random()
                < connection_prob
            ):

                g_index = random.randint(
                    0,
                    len(two_qb_list) - 1
                )

                gate_list.append(
                    {
                        "gate": two_qb_list[g_index],
                        "first_q": i,
                        "second_q": j
                    }
                )

    # ---------------------------------------------------------
    # Generate single-qubit gates
    # ---------------------------------------------------------

    n_one_qg = random.randint(
        0,
        2 * n_qubits
    )

    for _ in range(n_one_qg):

        q = random.randint(
            0,
            n_qubits - 1
        )

        g_index = random.randint(
            0,
            len(one_qb_list) - 1
        )

        gate_list.append(
            {
                "gate": one_qb_list[g_index],
                "first_q": q
            }
        )

    random.shuffle(
        gate_list
    )

    # ---------------------------------------------------------
    # Quantum circuit
    # ---------------------------------------------------------

    circuit = QuantumCircuit(
        n_qubits
    )

    data = [
        Parameter(f"p{i}")
        for i in range(n_qubits)
    ]

    # ---------------------------------------------------------
    # Rotational encoding
    # ---------------------------------------------------------

    if (
        activation
        == constants.QuanvActivation.FULL.value
    ):

        for q in range(n_qubits):

            circuit.rx(
                data[q] * 2 * math.pi,
                q
            )

    elif (
        activation
        == constants.QuanvActivation.HALF.value
    ):

        for q in range(n_qubits):

            circuit.rx(
                data[q] * math.pi,
                q
            )

    else:

        raise ValueError(
            f"Activation '{activation}' not supported "
            "for rotational encoding"
        )

    # ---------------------------------------------------------
    # Processing circuit
    # ---------------------------------------------------------

    for gate in gate_list:

        theta = (
            random.random()
            * math.pi
        )

        if gate["gate"] == "Cnot":

            circuit.cx(
                gate["first_q"],
                gate["second_q"]
            )

        elif gate["gate"] == "Swap":

            circuit.swap(
                gate["first_q"],
                gate["second_q"]
            )

        elif gate["gate"] == "SqrtSwap":

            circuit.rxx(
                math.pi / 2,
                gate["first_q"],
                gate["second_q"]
            )

            circuit.ryy(
                math.pi / 2,
                gate["first_q"],
                gate["second_q"]
            )

        elif gate["gate"] == "RX":

            circuit.rx(
                theta,
                gate["first_q"]
            )

        elif gate["gate"] == "RY":

            circuit.ry(
                theta,
                gate["first_q"]
            )

        elif gate["gate"] == "RZ":

            circuit.rz(
                theta,
                gate["first_q"]
            )

        elif gate["gate"] == "P":

            circuit.p(
                math.pi / 2,
                gate["first_q"]
            )

        elif gate["gate"] == "T":

            circuit.t(
                gate["first_q"]
            )

        elif gate["gate"] == "H":

            circuit.h(
                gate["first_q"]
            )

    return circuit