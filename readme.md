# MXFP4 Training Support Codebase

This repository provides the core implementation and framework integration for **MXFP4-based training**, accompanying the experimental results reported in the paper.

## Repository Structure

- **`kernel/`**  
  Contains the MXFP4 low-level implementation, including:
  - MXFP4 API wrapper interfaces
  - Core MXFP4 compute and communication kernels

- **`megatron-0.15.0rc7.diff`**  
  A patch file containing MXFP4-related modifications based on **Megatron-LM v0.15.0rc7**, enabling MXFP4 support in the training pipeline.

- **`transformer_engine-2.8/`**  
  Source-level modifications to **NVIDIA Transformer Engine v2.8**, extending the framework to support MXFP4 precision.

- **`deep_ep-1.2.1.diff`**  
  A patch file with MXFP4-related modifications for **DEEP-EP v1.2.1**, enabling MXFP4 support in expert parallel communication.

## Usage Instructions

1. **Compile MXFP4 Kernels**  
   Build the custom MXFP4 operators located in the `kernel/` directory according to your CUDA and system environment.

2. **Integrate Framework Modifications**  
   - Apply `megatron-0.15.0rc7.diff` to a clean Megatron-LM v0.15.0rc7 codebase.
   - Merge the modified components in `transformer_engine-2.8/` into the corresponding Transformer Engine v2.8 source tree.
   - Apply `deep_ep-1.2.1.diff` to a clean **DEEP-EP v1.2.1** codebase.

3. **Build and Run**  
   After integrating all framework modifications, rebuild the corresponding components and proceed with MXFP4-enabled training.

## Notes

- This codebase is intended for **research and experimental use**.
- The implementation and integration follow the design and evaluation methodology described in the paper.
