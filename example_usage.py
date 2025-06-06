#!/usr/bin/env python3
"""
Example usage script for TWISTER with new features:
1. Manual seed setting
2. Custom wandb project and entity
3. Checkpoint saving control
4. Media logging control  
5. Seed-based grouping and naming in wandb
6. Seed-based callback paths
"""

import subprocess
import sys

def run_training_example():
    """Example training with all new features enabled"""
    
    # Environment variables
    env_vars = {
        "env_name": "atari100k-alien",
        "run_name": "example_run"
    }
    
    # Command line arguments for training
    cmd = [
        sys.executable, "main.py",
        "--wandb",                                    # Enable wandb logging
        "--wandb_project", "my_custom_project",       # Custom project name
        "--wandb_entity", "my_username",              # Custom wandb username
        "--seed", "42",                               # Set manual seed
        "--save_checkpoints",                         # Enable checkpoint saving
        "--log_media",                                # Enable media logging
        "--mode", "training"
    ]
    
    # Set environment variables and run
    env = dict(os.environ, **env_vars)
    print("Running training example with custom settings...")
    print("Command:", " ".join(cmd))
    print("Environment variables:", env_vars)
    
    result = subprocess.run(cmd, env=env)
    return result.returncode

def run_multiple_seeds_example():
    """Example running the same experiment with different seeds"""
    
    seeds = [42, 123, 456]
    env_vars = {
        "env_name": "dmc-Cartpole-swingup", 
        "run_name": "seed_comparison"
    }
    
    for seed in seeds:
        cmd = [
            sys.executable, "main.py",
            "--wandb",
            "--wandb_project", "seed_comparison_project",
            "--wandb_entity", "my_username", 
            "--seed", str(seed),
            "--save_checkpoints",
            # Note: --log_media is NOT included to disable media logging
            "--mode", "training"
        ]
        
        env = dict(os.environ, **env_vars)
        print(f"\nRunning with seed {seed}...")
        print("Command:", " ".join(cmd))
        
        # In a real scenario, you might want to run these in parallel or queue them
        result = subprocess.run(cmd, env=env)
        if result.returncode != 0:
            print(f"Training failed for seed {seed}")
            return result.returncode
    
    print("\nAll seed runs completed!")
    return 0

def run_evaluation_example():
    """Example evaluation without saving anything"""
    
    env_vars = {
        "env_name": "atari100k-alien",
        "run_name": "example_run"
    }
    
    cmd = [
        sys.executable, "main.py",
        "--seed", "42",                               # Use same seed as training
        "--load_last",                                # Load last checkpoint
        "--mode", "evaluation",
        # Note: No --save_checkpoints, --log_media, or --wandb for evaluation
    ]
    
    env = dict(os.environ, **env_vars)
    print("Running evaluation example...")
    print("Command:", " ".join(cmd))
    
    result = subprocess.run(cmd, env=env)
    return result.returncode

if __name__ == "__main__":
    import os
    
    print("TWISTER Enhanced Usage Examples")
    print("=" * 50)
    
    print("\n1. Training with custom settings")
    print("-" * 30)
    # Uncomment to run training example
    # run_training_example()
    
    print("\n2. Multiple seed comparison (would create wandb group)")
    print("-" * 50)
    # Uncomment to run multiple seeds example  
    # run_multiple_seeds_example()
    
    print("\n3. Evaluation example")
    print("-" * 20)
    # Uncomment to run evaluation example
    # run_evaluation_example()
    
    print("\nFeature Summary:")
    print("=" * 50)
    print("✓ Manual seed setting: --seed <number>")
    print("✓ Custom wandb project: --wandb_project <name>") 
    print("✓ Custom wandb entity: --wandb_entity <username>")
    print("✓ Checkpoint saving control: --save_checkpoints (disabled by default)")
    print("✓ Media logging control: --log_media (disabled by default)")
    print("✓ Automatic wandb grouping by experiment name + seed-based run names")
    print("✓ Seed-specific callback paths for separate logging")
    print("\nCallback structure with seed:")
    print("callbacks/<run_name>/<env_name>/seed_<seed>/")
    print("\nWandb naming with seed:")
    print("Group: <run_name>/<env_name>")
    print("Run name: seed_<seed>_<run_name>/<env_name>") 