# Integrated Encoding to Enhance Quantum Convolutional Neural Networks



**Authors:**  
- Daniele Lizzio Bosco           :abacus: :dna: :envelope: 
- Beatrice Portelli  :abacus: :dna: 
- Giuseppe Serra   :abacus:




:abacus: Department of Mathematics, Computer Science and Physics, University of Udine, Udine, Italy 

:dna:  Department of Biology, University of Naples Federico II, Napoli, Italy

:envelope: Corresponding author [lizziobosco.daniele@spes.uniud.it]

**Read the paper here:**  
https://arxiv.org/abs/2410.05777

## Table of Contents
1. [Introduction](#Introduction)
2. [Reproducing the Experiments](#reproducing-the-experiments)
3. [Code Highlights](#code-highlights)
4. [Citation](#citation)


## Introduction

Quantum Machine Learning (QML) is a rapidly growing field that merges the principles of quantum computing with machine learning to tackle complex computational tasks. In the context of image processing, Quanvolutional Neural Networks (QuanvNNs) represent a promising approach, especially for Noisy Intermediate-Scale Quantum (NISQ) devices, which are quantum computers available in the near future with limited qubits and no error correction.

This repository implements an enhanced Quanvolutional Neural Network model, focusing on two major improvements: integrated encoding and flexible quantization. Our method addresses the limitations of existing QuanvNN architectures by introducing a more resource-efficient encoding scheme that combines encoding and processing into a single quantum circuit. Additionally, our flexible quantization method allows the dynamic adjustment of quantization levels, enabling users to balance computational efficiency with information retention.

We demonstrate the effectiveness of our model by comparing it with classical convolutional neural networks (CNNs) and traditional QuanvNNs using rotational encoding. This repository includes the code for implementing the proposed enhancements and reproducing the experiments on benchmark datasets.

## Reproducing the Experiments
1. **Requirements:**  
    This project is based on Python ```3.10.12```.
      <details>
    <summary>Requirements list</summary>
    
    ```yaml
    matplotlib==3.9.2
    numpy==2.1.0
    pandas==2.2.2
    Pillow==10.4.0
    PyYAML==6.0.1
    PyYAML==6.0.2
    qiskit==1.1.0
    qiskit_aer==0.14.1
    qiskit_aer_gpu==0.14.1
    qiskit_ibm_runtime==0.24.0
    scipy==1.14.1
    skimage==0.0
    torch==2.3.0
    torchvision==0.18.0
    tqdm==4.66.4

    ```

    </details>
2. **Setup Instructions:**  
    To set up the environment and install necessary dependencies:

   ```sh
   git clone https://github.com/Dan-LB/integrated_encoding_for_QuanvNN.git
   cd integrated_encoding_for_QuanvNN 
   pip install -r requirements.txt
   ```

3. **Main results:**
    The following steps can be performed to obtain the main results - classification accuracy of the proposed integrated encoding, compared to rotational encoding and classical CNN, corresponding to Table 3 and Table 4. 

    By default, the scripts perform the experiments on the MiraBest dataset. To use the LArTPC dataset instead, switch ```SELECTED_TASK = "MiraBest"``` to ```SELECTED_TASK = "LArTPC"``` in the considered files.

    

    1. Execute ```main.py```. This script checks all the models in the ```configs``` folder and runs the experiment for 10 different seeds.
    2. Execute ```process_results.py```.
    3. The results will be stored in the folder ```results```.

4. **Other results:**
    - Fig. 3 can be obtained from the notebook ```Compute_quantization_error.ipynb```.
    - Fig. 4 can be obtained from the notebook ```Compute_quantization_reduction.ipynb```.
    - The procedure to produce Fig. 7 is described in the notebook ```Compute_expressibility.ipynb```.

## Code Highlights

* #### ```Quanvolutional_Layer.py```
    This file contains the class ```QuanvolutionalLayer``` as a Torch.nn module. 
    Once initialized, the layer generates the circuits to be used as filters.
* #### ```circuit_builder.py```
    Contains the functions to construct the circuits, with both the standard approach and the proposed one.
* #### ```model_builder.py```
    Contains the function to construct a Quanvolutional model by constructing first a classical CNN, and then stacking a Quanvolutional layer on it with the function ```quanv_model = stack_quanv_on_top(quanv_layer, classical_model)```. 
* #### ```main.py```
    This script is used to perform all the experiments with 10 different seeds for each model present in the ```config``` folder.
* #### ```configs```
    The ```configs``` folder contains all the configuration used in this work.

    Example of $\text{QNN-Int-Simple}$:
    <details>
    <summary><i>QNN-Int-Simple-k3.yaml</i></summary>
    
    ```yaml

    encoding: INTEGRATED
    model:
    conv1:
        in_channels: 1
        kernel_size: 3
        out_channels: 16
        padding: 0
    dropout_conv_rate: 0.2
    dropout_fc_rate: 0.2
    fc1:
        out_features: 32
    fc2:
        out_features: 2
    input_shape:
    - 1
    - 30
    - 30
    quanv:
    L: 18
    activation: Full
    kernel_size: 3
    n_qubits: 4
    n_shots: 1000
    ```

    </details>
    


## Citation
```bibtex
@ARTICLE{11303607,
  author={Lizzio Bosco, Daniele and Portelli, Beatrice and Serra, Giuseppe},
  journal={IEEE Transactions on Quantum Engineering}, 
  title={Integrated Encoding and Quantization to Enhance Quanvolutional Neural Networks}, 
  year={2025},
  volume={},
  number={},
  pages={1-20},
  keywords={Encoding;Quantization (signal);Integrated circuit modeling;Qubit;Image coding;Quantum circuit;Neural networks;Quantum state;Logic gates;Convolutional neural networks;Convolutional neural networks;image processing;NISQ;quantum computing;quantum encoding;quantum machine learning;quanvolutional neural network},
  doi={10.1109/TQE.2025.3646040}}

```

## Local Paper-Baseline Setup

This project was imported from the authors' repository at:

https://github.com/Dan-LB/integrated_encoding_for_QuanvNN

The imported revision is `0466647`. The `paper-baseline` branch preserves that
revision and must remain unchanged. The `improved-model` branch is reserved for
future research changes. No training or analysis has been run during setup.

### Environment

The authors specify Python `3.10.12`. Install the exact package versions from
`requirements.txt`; do not upgrade them for baseline work. The requirements
include the authors' versions of Qiskit `1.1.0`, Qiskit Aer `0.14.1`, PyTorch
`2.3.0`, torchvision `0.18.0`, NumPy `2.1.0`, SciPy `1.14.1`, Matplotlib
`3.9.2`, PyYAML `6.0.1`, pandas `2.2.2`, Pillow `10.4.0`, scikit-image
`0.24.0`, and tqdm `4.66.4`.

Create and activate a Python 3.10.12 environment, then install the pinned
requirements:

```powershell
py -3.10 -m venv venv
.\venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

The commands above are setup instructions only; they were not run as part of
this import.

### MiraBest Data

`utils/get_dataset/dataset_Mirabest.py` calls `MiraBest(root='./dataMirabest',
train=..., download=True, ...)`. The expected source is the authors' archive
`MiraBest_basic_batches.tar.gz`, containing `dataMirabest/batches/` with the
pickle files `data_batch_1` through `data_batch_9`, `test_batch`, and
`batches.meta`. The loader expects each batch to contain image data and labels,
and verifies the authors' MD5 checksums.

The loader reads 150×150 single-channel images, converts them to grayscale PIL
images, converts them to tensors, and resizes them to 30×30. The model input
shape is therefore `(1, 30, 30)` and MiraBest is treated as a two-class
classification task. The repository does not contain the dataset, and no data
was fabricated or downloaded during setup.

### Original Experiment Matrix

The existing `configs/` directory already contains the complete matrix:

- `CNN.yaml`
- `QNN-Rot-k2.yaml`, `QNN-Rot-k3.yaml`, `QNN-Rot-k4.yaml`
- `QNN-Int-Simple-k2.yaml` through `QNN-Int-Simple-k5.yaml`
- `QNN-Int-RndMul-k2.yaml` through `QNN-Int-RndMul-k5.yaml`
- `QNN-Int-RndLin-k2.yaml` through `QNN-Int-RndLin-k5.yaml`

`main.py` uses MiraBest by default, quantization level `50`, eight quanvolution
channels, ten seeds, and 1000 epochs. It iterates over every YAML file in
`configs/`, so the original command runs the full matrix rather than a single
model:

```powershell
python main.py
python process_results.py
```

These commands have not been run. Original output is written below `exps/` and
processed summaries below `results/`; no baseline accuracy, runtime, warning,
or checkpoint result is currently available.

### Activation Parameter

`constants.py` defines the supported activation names. In the integrated
builder, `activation` controls the angle applied to each encoded pixel value:
`Full` uses `2*pi*beta*x`, `Half` uses `pi*beta*x`, `Shifted` adds `pi/2`,
`Random` adds a random phase, and `Fixed` uses `pi*x`. In the rotational
builder, the original implementation supports `Full` and `Half`.

The checked-in paper configurations use `Fixed` for Int-Simple, `Full` for
Int-RndMul, `Random` for Int-RndLin, and `Half` for QNN-Rot. The setting is in
each YAML file's `quanv.activation` field and is part of the circuit encoding
definition, not a classical neural-network activation function.

### Important Files

- `main.py`: original multi-seed training and quanvolution preprocessing driver.
- `quanvs/Quanvolutional_Layer.py`: original Torch quanvolutional layer and patch lookup.
- `quanvs/circuit_builder.py`: original integrated and rotational circuit builders.
- `quanvs/model_builder.py`: original classical CNN and model composition.
- `constants.py`: original enums, including encodings and activation names.
- `utils/get_dataset/dataset_Mirabest.py` and `data/MiraBest.py`: original MiraBest loader.
- `utils/train_and_test.py`: original training and test loops.
- `process_results.py`: original aggregation and CSV generation.
- `Compute_quantization_error.ipynb`, `Compute_quantization_reduction.ipynb`, and `Compute_expressibility.ipynb`: author-provided analyses.

These files and the YAML configurations belong to the paper implementation
and should remain unchanged for baseline work. Future modifications belong on
`improved-model`, with experiment outputs separated as
`results/paper_baseline/` and `results/improved_model/`.

The repository contains no patch-distribution/PCA analysis. That would be new
analysis and must be kept separate from the author-provided notebooks.

