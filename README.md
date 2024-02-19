# Automated Identification of Prospects from XRD Data

### Abstract

New materials are frequently synthesized and optimized with the explicit intention to improve their properties to meet the ever-increasing societal requirements for high-performance and energy-efficient electronics, new battery concepts, better recyclability, and low-energy manufacturing processes. This often involves exploring vast combinations of stoichiometries and compositions, a process made more efficient by high-throughput robotic platforms. Nonetheless, subsequent analytical methods are essential to screen the numerous samples and identify promising material candidates. X-ray diffraction is a commonly used analysis method available in most laboratories which gives insight into the crystalline structure and reveals the presence of phases in a powder sample. Herein, a method for automating the analysis of XRD patterns, which uses a neural network model to classify samples into nondiffracting, single-phase, and multi-phase structures, is presented. To train neural networks for identifying materials with compositions not matching known crystallographic structures, a synthetic data generation approach is developed. The application of the neural networks on high-entropy oxides experimental data is demonstrated, where materials frequently deviate from anticipated structures. Our approach, not limited to these materials, seamlessly integrates into high-throughput data analysis pipelines, either filtering acquired patterns or serving as a standalone method for automated material exploration workflows.

[Full Publication](https://onlinelibrary.wiley.com/doi/10.1002/aisy.202300501)

### Repository

This repository provides the basic functionality to train a neural network model for automated identifying materials based on their characteristic XRD patterns. The target structure is defined through a Crystallographic Information File (cif), based on which artificial XRD patterns are generated. The [*python-powder-diffraction*](https://github.com/jschuetzke/python-powder-diffraction) package is used to simulate realistic XRD patterns. These generated patterns include positive (pure target material) and negative examples (containing impurities/alternative patterns), which are used to train a neural network model. For the integration of the model, the TensorFlow library is integrated. Once the model is trained, it can be applied to measured XRD patterns for identification of the target structure.

### Usage

1. Clone the repository and setup a new environment (e.g., using conda).
2. Setup the environment using the *requirements.txt* file `pip install -r requirements.txt`
3. Obtain or define a cif that describes the target structure
4. Modify the config.yaml file according to the use-case (measurement modalities)
5. Run *generate_training_data.py* and provide a unique specifier for this target material
6. Run *train.py* to train the neural network model for this application
7. Use the resulting model.h5 file for automated analysis of measured XRD data

### Example

```
python generate_training_data.py example --file_path test.cif
python train.py example
```
