import torch
import gc
import torch
import os

def print_gpu_memory_usage(stage_name=""):
    """Print detailed GPU memory usage for all available GPUs"""
    print(f"\n{'='*60}")
    print(f"GPU Memory Usage - {stage_name}")
    print(f"{'='*60}")
    
    for i in range(torch.cuda.device_count()):
        print(f"\n--- GPU {i} ---")
        allocated = torch.cuda.memory_allocated(i) / 1024**3
        reserved = torch.cuda.memory_reserved(i) / 1024**3
        max_allocated = torch.cuda.max_memory_allocated(i) / 1024**3
        total = torch.cuda.get_device_properties(i).total_memory / 1024**3
        
        print(f"  Allocated: {allocated:.2f} GB")
        print(f"  Reserved:  {reserved:.2f} GB")
        print(f"  Max Alloc: {max_allocated:.2f} GB")
        print(f"  Total:     {total:.2f} GB")
        print(f"  Free:      {total - reserved:.2f} GB")
    print(f"{'='*60}\n")

def print_model_device_map(model, model_name="Model"):
    """Print which device each model component is on"""
    print(f"\n{model_name} Device Map:")
    if hasattr(model, 'hf_device_map'):
        print(f"  HF Device Map: {model.hf_device_map}")
    
    for name, param in model.named_parameters():
        if param.device.type == 'cuda':
            print(f"  {name}: GPU {param.device.index}")
            break  # Just show first layer to avoid spam
    print(f"  Model device: {next(model.parameters()).device}\n")




def diagnose_gpu_setup():
    """Print GPU setup for debugging"""
    print("\n" + "="*80)
    print("GPU SETUP DIAGNOSIS")
    print("="*80)
    
    # What does CUDA_VISIBLE_DEVICES say?
    visible = os.environ.get('CUDA_VISIBLE_DEVICES', 'Not set')
    print(f"CUDA_VISIBLE_DEVICES: {visible}")
    
    # How many GPUs does PyTorch see?
    num_gpus = torch.cuda.device_count()
    print(f"PyTorch sees {num_gpus} GPUs")
    
    # List each GPU
    for i in range(num_gpus):
        props = torch.cuda.get_device_properties(i)
        print(f"  GPU {i}: {props.name} ({props.total_memory / 1024**3:.1f} GB)")
    
    # Accelerator info
    try:
        from accelerate import Accelerator
        acc = Accelerator()
        print(f"\nAccelerator:")
        print(f"  Device: {acc.device}")
        print(f"  Process index: {acc.process_index}")
        print(f"  Num processes: {acc.num_processes}")
        print(f"  Is main process: {acc.is_main_process}")
    except Exception as e:
        print(f"Could not get Accelerator info: {e}")
    
    print("="*80 + "\n")
