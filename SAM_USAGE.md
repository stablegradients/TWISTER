# SAM (Sharpness-Aware Minimization) Integration

This document describes how to use the SAM (Sharpness-Aware Minimization) optimizer that has been integrated into the TWISTER codebase.

## Overview

SAM is an optimization technique that seeks parameters that lie in neighborhoods having uniformly low loss. The key idea is to perturb the model parameters in the direction that increases loss the most, then compute gradients at this perturbed location to encourage the optimizer to find flatter minima.

## Usage

### Command Line Arguments

To use SAM, you can add the following command line arguments:

- `--use_sam`: Enable SAM optimization
- `--adaptive_sam`: Use the adaptive variant of SAM (default: False)
- `--sam_rho`: Set the perturbation radius (default: 0.05)

### Basic Usage

```bash
# Basic SAM usage
python main.py --config configs/your_config.py --use_sam

# SAM with adaptive variant
python main.py --config configs/your_config.py --use_sam --adaptive_sam

# SAM with custom perturbation radius
python main.py --config configs/your_config.py --use_sam --sam_rho 0.1

# Combined options
python main.py --config configs/your_config.py --use_sam --adaptive_sam --sam_rho 0.02
```

## How It Works

### Two-Step Optimization Process

When SAM is enabled, the training step follows this process:

1. **First Step**: 
   - Forward pass: `losses, metrics, truths, preds = self.forward_model(...)`
   - Backward pass and scaling/unscaling
   - SAM first step: perturbs parameters in the direction that increases loss

2. **Second Step**:
   - Forward pass again: `losses, metrics, truths, preds = self.forward_model(...)`
   - Backward pass and scaling/unscaling  
   - SAM second step: actual parameter update using gradients at perturbed location

### Parameters

- **rho**: The perturbation radius. Controls how far to perturb parameters. Typical values are 0.05-0.1
- **adaptive**: When True, uses parameter-wise perturbation scaling based on parameter magnitudes

## Examples

### Training with SAM
```bash
# Train with SAM enabled
python main.py -c configs/twister.py -m training --use_sam --wandb

# Train with adaptive SAM and custom rho
python main.py -c configs/twister.py -m training --use_sam --adaptive_sam --sam_rho 0.1 --wandb
```

### Environment Variables
You can also combine SAM with environment variables:
```bash
env_name=CartPole-v1 python main.py -c configs/twister.py -m training --use_sam --adaptive_sam
```

## Implementation Details

### Optimizer Wrapping
SAM wraps the existing optimizer (typically Adam). The base optimizer is used for the actual parameter updates, while SAM handles the perturbation logic.

### Mixed Precision Support
SAM is fully compatible with mixed precision training. The gradient scaling and unscaling operations are properly handled for both the first and second steps.

### Memory Considerations
SAM requires two forward passes per training step, which approximately doubles the memory usage and computation time. However, only one set of activations needs to be stored at a time.

### Gradient Accumulation
SAM works correctly with gradient accumulation. The two-step process is only executed when `acc_step >= accumulated_steps`.

## Configuration

The SAM optimizer is automatically configured when you use the command line flags. The system will:

1. Detect when `--use_sam` is specified
2. Wrap the existing optimizer (e.g., Adam) with SAM
3. Print a confirmation message: `"Using SAM optimizer (rho=0.05, adaptive=False) wrapping Adam"`
4. Execute the two-step optimization process during training

## Troubleshooting

### Common Issues

1. **Memory errors**: SAM requires more memory due to two forward passes. Consider reducing batch size.

2. **Slower training**: SAM approximately doubles training time due to the two-step process. This is expected.

3. **Checkpointing**: SAM state is properly saved and loaded with model checkpoints.

### Performance Tips

1. Start with the default `rho=0.05` and adjust based on your specific problem
2. Use `--adaptive_sam` for problems with parameters of very different scales
3. Monitor both training loss and validation metrics, as SAM aims for better generalization

## Technical Notes

- SAM is implemented as a wrapper around existing optimizers
- The implementation follows the original SAM paper's algorithm
- Both standard and adaptive variants are supported
- Full compatibility with the existing training pipeline, including mixed precision and gradient accumulation

## References

- Original SAM paper: "Sharpness-Aware Minimization for Efficiently Improving Generalization"
- Adaptive SAM: Uses parameter-wise perturbation scaling for better performance with diverse parameter scales 