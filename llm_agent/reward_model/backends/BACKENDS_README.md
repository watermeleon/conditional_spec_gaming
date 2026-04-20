# LLM Agent Reward with Flexible Backends

**NEW in v2.1**: Choose between vLLM server (production) or local model (testing/development)!

## 🎯 Backend Options

### Option 1: vLLM Server Backend (Production)
- ✅ Fast and scalable
- ✅ Best for production training
- ✅ Handles high concurrency
- ❌ Requires running vLLM server

### Option 2: Local Model Backend (Testing/Development)  
- ✅ No server needed
- ✅ Perfect for testing and debugging
- ✅ Works with any HuggingFace model
- ✅ Supports quantization (8-bit, 4-bit)
- ❌ Slower than vLLM server
- ❌ Limited concurrency

---

## 🚀 Quick Start

### Using vLLM Server (Production)

```python
from llm_agent_reward_v2 import LLMAgentRewardVLLM

# Option A: Provide vLLM URL (auto-creates VLLMBackend)
reward_func = LLMAgentRewardVLLM(
    vllm_server_url="http://localhost:8000/v1",
    judge_model_name="google/gemma-2-2b-it",
    llm_prompts=my_prompts
)
```

### Using Local Model (Testing)

```python
from llm_agent_reward_v2 import LLMAgentRewardVLLM

# Option B: Use local model (auto-creates LocalLLMBackend)
reward_func = LLMAgentRewardVLLM(
    use_local=True,
    judge_model_name="google/gemma-2-2b-it",  # Any HF model
    llm_prompts=my_prompts,
    
    # Optional: Optimize for your hardware
    local_device="cuda",           # "cuda", "cpu", or "auto"
    local_dtype="float16",          # "float16", "bfloat16", "float32"
    local_load_in_4bit=True,       # Use 4-bit quantization
)
```

### Advanced: Custom Backend

```python
from llm_agent_reward_v2 import LLMAgentRewardVLLM
from backends import LocalLLMBackend

# Create custom backend with specific settings
my_backend = LocalLLMBackend(
    model_name="Qwen/Qwen2.5-0.5B-Instruct",
    device="cuda",
    load_in_4bit=True
)

# Pass to reward function
reward_func = LLMAgentRewardVLLM(
    backend=my_backend,
    llm_prompts=my_prompts
)
```

---

## 📖 Complete Examples

### Example 1: Quick Testing with Small Local Model

```python
"""
Quick test setup - no vLLM server needed!
Perfect for debugging prompts and logic.
"""

from llm_agent_reward_v2 import LLMAgentRewardVLLM

# Use tiny model for fast testing
reward_func = LLMAgentRewardVLLM(
    use_local=True,
    judge_model_name="Qwen/Qwen2.5-0.5B-Instruct",  # Small, fast model
    llm_prompts=my_prompts,
    evaluation_strategy="helpful",
    local_device="cuda",
    local_load_in_4bit=True,  # Saves memory
    verbose=True  # See what's happening
)

# Test with a few examples
test_prompts = [[{"role": "user", "content": "Help me with Python"}]]
test_completions = [[{"role": "assistant", "content": "Sure! What do you need?"}]]

rewards = reward_func(test_completions, test_prompts)
print(f"Test rewards: {rewards}")
```

### Example 2: Development with Better Local Model

```python
"""
Development setup - good quality local model.
Use for iterating on training setup.
"""

from llm_agent_reward_v2 import LLMAgentRewardVLLM

reward_func = LLMAgentRewardVLLM(
    use_local=True,
    judge_model_name="google/gemma-2-2b-it",  # Better quality
    llm_prompts=my_prompts,
    evaluation_strategy="ethical",
    local_device="cuda",
    local_dtype="bfloat16",  # Good balance
    max_concurrent=1,  # Local models typically sequential
)

# Use in training
trainer = GRPOTrainer(
    model=model,
    reward_funcs=[reward_func],
    args=training_args,
    train_dataset=train_dataset,
)

trainer.train()
```

### Example 3: Production with vLLM Server

```python
"""
Production setup - fast vLLM server.
Use for actual training runs.
"""

from llm_agent_reward_v2 import LLMAgentRewardVLLM

# Start vLLM server first:
# vllm serve google/gemma-2-2b-it --port 8000

reward_func = LLMAgentRewardVLLM(
    vllm_server_url="http://localhost:8000/v1",
    judge_model_name="google/gemma-2-2b-it",
    llm_prompts=my_prompts,
    evaluation_strategy="ethical",
    max_concurrent=32,  # High concurrency with vLLM
    log_rewards_steps=10,
)

# Train with full dataset
trainer = GRPOTrainer(
    model=model,
    reward_funcs=[reward_func],
    args=training_args,
    train_dataset=full_dataset,
)

trainer.train()
```

### Example 4: Switching Between Backends

```python
"""
Easy to switch between local and vLLM!
Same code, just change one parameter.
"""

# Testing mode
TESTING = True

reward_func = LLMAgentRewardVLLM(
    # Switch based on mode
    use_local=TESTING,
    vllm_server_url=None if TESTING else "http://localhost:8000/v1",
    
    # Common settings
    judge_model_name="google/gemma-2-2b-it",
    llm_prompts=my_prompts,
    evaluation_strategy="ethical",
    
    # Local-specific (ignored if vLLM)
    local_load_in_4bit=True,
    local_device="cuda",
)

# Rest of code stays the same!
```

---

## 🔧 Parameter Reference

### Backend Selection (choose ONE)

```python
LLMAgentRewardVLLM(
    # Option 1: Custom backend instance
    backend=my_backend,  # Pre-configured LLMBackend
    
    # Option 2: vLLM server URL
    vllm_server_url="http://localhost:8000/v1",
    
    # Option 3: Local model flag
    use_local=True,
    
    # ... other parameters
)
```

### Local Backend Configuration

```python
LLMAgentRewardVLLM(
    use_local=True,
    
    # Device selection
    local_device="auto",  # "auto", "cuda", "cpu", "cuda:0", etc.
    
    # Precision
    local_dtype="auto",  # "auto", "float16", "bfloat16", "float32"
    
    # Quantization (saves memory)
    local_load_in_8bit=False,  # 8-bit quantization
    local_load_in_4bit=False,  # 4-bit quantization (most memory efficient)
    
    # Concurrency (usually 1 for local)
    max_concurrent=1,
)
```

### Recommended Local Model Settings

```python
# For testing (fastest)
LLMAgentRewardVLLM(
    use_local=True,
    judge_model_name="Qwen/Qwen2.5-0.5B-Instruct",  # Tiny model
    local_load_in_4bit=True,
)

# For development (balanced)
LLMAgentRewardVLLM(
    use_local=True,
    judge_model_name="google/gemma-2-2b-it",  # Medium quality
    local_dtype="float16",
)

# For validation (best quality)
LLMAgentRewardVLLM(
    use_local=True,
    judge_model_name="Qwen/Qwen2.5-7B-Instruct",  # Better quality
    local_load_in_8bit=True,  # If you have enough VRAM
)
```

---

## 💻 Hardware Requirements

### Local Backend Memory Usage

| Model Size | Precision | Approximate VRAM |
|------------|-----------|------------------|
| 0.5B params | 4-bit | ~0.5 GB |
| 0.5B params | float16 | ~1 GB |
| 2B params | 4-bit | ~1.5 GB |
| 2B params | float16 | ~4 GB |
| 7B params | 4-bit | ~4 GB |
| 7B params | 8-bit | ~7 GB |
| 7B params | float16 | ~14 GB |

**Tip**: Use 4-bit quantization for testing to minimize VRAM usage!

---

## 🔄 Migration Guide

### From v2.0 (vLLM only) to v2.1 (Flexible backends)

**Old code (still works!):**
```python
reward_func = LLMAgentRewardVLLM(
    vllm_server_url="http://localhost:8000/v1",
    judge_model_name="google/gemma-2-2b-it",
)
```

**New code (same result):**
```python
# Explicit - same as before
reward_func = LLMAgentRewardVLLM(
    vllm_server_url="http://localhost:8000/v1",  # Creates VLLMBackend
    judge_model_name="google/gemma-2-2b-it",
)

# OR use local for testing
reward_func = LLMAgentRewardVLLM(
    use_local=True,  # Creates LocalLLMBackend
    judge_model_name="google/gemma-2-2b-it",
)
```

---

## 🧪 Testing Workflow

Recommended development workflow:

1. **Prototype** with local small model (Qwen 0.5B)
   - Test prompts
   - Debug logic
   - Verify reward calculation
   
2. **Develop** with local medium model (Gemma 2B)
   - Test with more samples
   - Validate prompt quality
   - Check edge cases

3. **Validate** with local large model (Qwen 7B) or vLLM
   - Final prompt testing
   - Quality check on small dataset

4. **Train** with vLLM server
   - Full training run
   - Production scale

---

## 🐛 Troubleshooting

### Local Backend Issues

**Issue: Out of memory**
```python
# Solution: Use quantization
reward_func = LLMAgentRewardVLLM(
    use_local=True,
    local_load_in_4bit=True,  # Reduces memory by 4x
)
```

**Issue: Model loading is slow**
```python
# Solution: Model is downloaded first time only
# Subsequent runs are faster
# Or pre-download: huggingface-cli download MODEL_NAME
```

**Issue: Scores always 0.0**
```python
# Solution: Check prompt format
reward_func = LLMAgentRewardVLLM(
    use_local=True,
    verbose=True,  # See what model generates
)
# Check generated text matches expected format (single number)
```

### Backend Selection Issues

**Issue: Backend not specified**
```
ValueError: Must provide either backend, vllm_server_url, or use_local=True
```

```python
# Solution: Choose ONE backend option
reward_func = LLMAgentRewardVLLM(
    use_local=True,  # Or vllm_server_url=...
    # ...
)
```

---

## 📊 Performance Comparison

| Backend | Speed | Concurrency | Memory | Setup |
|---------|-------|-------------|--------|-------|
| vLLM Server | ⚡⚡⚡ Very Fast | ✅ High (32+) | 🟢 Server only | Requires server |
| Local 0.5B | ⚡⚡ Fast | ⚠️ Low (1) | 🟢 Low (~1GB) | Instant |
| Local 2B | ⚡ Medium | ⚠️ Low (1) | 🟡 Medium (~4GB) | Instant |
| Local 7B | 🐌 Slow | ⚠️ Low (1) | 🔴 High (~14GB) | Instant |

**Recommendation**: 
- Development: Local 0.5B or 2B with 4-bit quantization
- Production: vLLM server

---

## 🎯 Summary

**Key Benefits:**
- ✅ No server needed for testing
- ✅ Same API for both backends
- ✅ Easy to switch between modes
- ✅ Quantization support for local models
- ✅ Backward compatible with v2.0

**When to use each:**
- **Local**: Quick tests, prompt debugging, no server available
- **vLLM**: Production training, large datasets, high performance

---

## 📝 Complete Example Script

```python
#!/usr/bin/env python3
"""
Complete example showing backend flexibility.
"""

from llm_agent_reward_v2 import LLMAgentRewardVLLM
from trl import GRPOTrainer, GRPOConfig

# Configuration
TESTING_MODE = True  # Switch to False for production

# Define prompts (same for both backends)
prompts = {
    "helpful_evaluation": (
        "Rate how helpful (1-10) this response is:\n{agent_message}\n\n"
        "Respond with only a number 1-10:"
    ),
    "user_persona": {"name": "User"},
}

# Create reward function (backend chosen automatically)
reward_func = LLMAgentRewardVLLM(
    # Backend selection
    use_local=TESTING_MODE,
    vllm_server_url=None if TESTING_MODE else "http://localhost:8000/v1",
    
    # Model
    judge_model_name="google/gemma-2-2b-it",
    
    # Prompts and strategy
    llm_prompts=prompts,
    evaluation_strategy="helpful",
    
    # Local backend optimization
    local_load_in_4bit=True if TESTING_MODE else False,
    local_device="cuda",
    
    # Behavior
    verbose=True,
    max_concurrent=1 if TESTING_MODE else 32,
)

# Use in training (same code for both backends!)
training_args = GRPOConfig(
    output_dir="./output",
    max_steps=10 if TESTING_MODE else 1000,
)

trainer = GRPOTrainer(
    model="Qwen/Qwen2.5-0.5B-Instruct",
    reward_funcs=[reward_func],
    args=training_args,
    train_dataset=dataset,
)

trainer.train()

# Clean up
reward_func.cleanup()
```

**That's it!** Switch `TESTING_MODE` to change backends. 🎉
