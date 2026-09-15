# Data-Aware Resource-Adaptive Quanvolutional Neural Network

A research implementation extending the **Integrated Encoding and Quantization** approach for Quanvolutional Neural Networks (QuanvNNs).

## Objective

The main goal is to make quantum feature extraction more **data-aware and resource-efficient** instead of using the same resource configuration for every image patch.

The current framework focuses on:

- Reducing unnecessary quantum circuit executions
- Reusing previously computed quantum outputs
- Selecting better and less-redundant quantum filters
- Allocating circuit resources according to feature importance
- Adjusting measurement shots according to quantization level

## Our Five Innovations

### 1. Adaptive Quantization
Patch complexity is analyzed and the quantization level is selected from:

`N = 10, 25, 50`

### 2. Similarity-Aware Memoization
Before executing a quantum circuit:

- Exact matches are reused directly.
- Similar patches are checked using similarity matching and validation.
- Only unmatched patches are sent to the quantum circuit.

### 3. Diversity-Aware Filter Selection
Multiple candidate quantum filters are evaluated using training data.

The final 8 filters are selected based on:

- Quality
- Diversity

### 4. Information-Aware Gate Allocation
Feature importance is estimated from training data and used to distribute the fixed quantum gate budget.

For `k = 3`:

`L = 18 gates`

Every feature remains represented.

### 5. Adaptive Measurement Shots
The number of measurement shots depends on the quantization level:

| Quantization | Shots |
|---|---:|
| N = 10 | 250 |
| N = 25 | 500 |
| N = 50 | 1000 |

## Overall Pipeline

```text
Input Image
    ↓
3×3 Patch Extraction
    ↓
Local Patch Analysis
    ↓
Adaptive Quantization
    ↓
Exact / Similarity Memoization
    ↓
Quantum Filter Selection
    ↓
Information-Aware Gate Allocation
    ↓
Adaptive Measurement
    ↓
Quantum Feature Maps
    ↓
Classical CNN
    ↓
Binary Classification
