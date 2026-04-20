#!/usr/bin/env python3
"""
Simple test script to verify both backends work.
Run this to test your setup!
"""

import asyncio
from llm_agent.reward_model.llm_agent_reward import LLMAgentRewardVLLM

# Simple test prompts
test_prompts = {
    "default_evaluation": (
        "Rate the helpfulness (0-10) of this response:\n"
        "{agent_message}\n\n"
        "Respond with only a number 0-10:"
    ),
    "user_persona": {"name": "User"},
}

def test_backend(backend_name, reward_func):
    """Test a reward function with sample data."""
    print(f"\n{'='*60}")
    print(f"Testing {backend_name}")
    print('='*60)
    
    # Mock TRL format data
    test_contexts = [
        "Help me learn Python",
        "What's the capital of France?",
    ]
    
    test_responses = [
        "I'd be happy to help you learn Python! Let's start with basics.",
        "The capital of France is Paris.",
    ]
    
    # Convert to TRL format
    prompts_trl = [
        [{"role": "user", "content": ctx}] 
        for ctx in test_contexts
    ]
    
    completions_trl = [
        [{"role": "assistant", "content": resp}]
        for resp in test_responses
    ]
    
    # Calculate rewards
    print("\nCalculating rewards...")
    rewards = reward_func(completions_trl, prompts_trl)
    
    # Display results
    print(f"\nResults:")
    for i, (ctx, resp, reward) in enumerate(zip(test_contexts, test_responses, rewards)):
        print(f"\n  Example {i+1}:")
        print(f"    Context: {ctx}")
        print(f"    Response: {resp}")
        print(f"    Reward: {reward:.2f}")
    
    print(f"\n✓ {backend_name} test completed!")
    return rewards


def main():
    """Run tests for both backends."""
    print("🧪 Backend Testing Script")
    print("This will test both vLLM and local backends (if available)")
    
    # Test 1: Local Backend (always available)
    print("\n" + "="*60)
    print("TEST 1: Local Backend (No server needed)")
    print("="*60)
    
    try:
        print("\nLoading local model (this may take a moment)...")
        local_reward = LLMAgentRewardVLLM(
            use_local=True,
            judge_model_name="Qwen/Qwen2.5-0.5B-Instruct",  # Small model
            llm_prompts=test_prompts,
            evaluation_strategy="default",
            local_device="cpu",  # Use CPU for compatibility
            local_load_in_4bit=False,  # 4-bit requires GPU
            verbose=False,
        )
        
        local_rewards = test_backend("Local Backend", local_reward)
        local_reward.cleanup()
        
        print("\n✅ Local backend works!")
        
    except Exception as e:
        print(f"\n❌ Local backend failed: {e}")
        print("Make sure transformers and torch are installed:")
        print("  pip install transformers torch")
        return
    
    # Test 2: vLLM Backend (if server is running)
    print("\n" + "="*60)
    print("TEST 2: vLLM Backend (Requires running server)")
    print("="*60)
    
    # Check if user wants to test vLLM
    try:
        response = input("\nDo you have a vLLM server running? (y/N): ").strip().lower()
        if response == 'y':
            server_url = input("Enter vLLM server URL [http://localhost:8000/v1]: ").strip()
            if not server_url:
                server_url = "http://localhost:8000/v1"
            
            model_name = input("Enter model name [google/gemma-2-2b-it]: ").strip()
            if not model_name:
                model_name = "google/gemma-2-2b-it"
            
            print(f"\nConnecting to vLLM server at {server_url}...")
            vllm_reward = LLMAgentRewardVLLM(
                vllm_server_url=server_url,
                judge_model_name=model_name,
                llm_prompts=test_prompts,
                evaluation_strategy="default",
                max_concurrent=4,
                verbose=False,
            )
            
            vllm_rewards = test_backend("vLLM Backend", vllm_reward)
            vllm_reward.cleanup()
            
            print("\n✅ vLLM backend works!")
            
            # Compare results
            print("\n" + "="*60)
            print("Comparison")
            print("="*60)
            print(f"\nLocal rewards: {[f'{r:.2f}' for r in local_rewards]}")
            print(f"vLLM rewards:  {[f'{r:.2f}' for r in vllm_rewards]}")
            print("\nNote: Small differences are normal due to different implementations")
            
        else:
            print("\n⏭️  Skipping vLLM test")
            print("To test vLLM backend later, start a server:")
            print("  vllm serve google/gemma-2-2b-it --port 8000")
            
    except KeyboardInterrupt:
        print("\n\n⏭️  vLLM test cancelled")
    except Exception as e:
        print(f"\n❌ vLLM backend failed: {e}")
        print("Make sure vLLM server is running and accessible")
    
    # Summary
    print("\n" + "="*60)
    print("✅ Testing Complete!")
    print("="*60)
    print("\nYou can now use either backend in your training:")
    print("\n  # Local (for testing)")
    print("  reward = LLMAgentRewardVLLM(use_local=True, ...)")
    print("\n  # vLLM (for production)")
    print("  reward = LLMAgentRewardVLLM(vllm_server_url='...', ...)")
    print("\nSee BACKENDS_README.md for more examples!")


if __name__ == "__main__":
    main()
