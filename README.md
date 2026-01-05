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

