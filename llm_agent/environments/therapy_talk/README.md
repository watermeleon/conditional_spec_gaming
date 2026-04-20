# Therapy Talk Dataset Loader

## Overview

The `therapy_talk_loader.py` creates datasets for training LLM agents in a therapy chatbot scenario. The key architectural principle is the **separation of agent and judge contexts**: the agent being trained sees its instructions, while the reward model (judge) evaluates responses without seeing those instructions.

## Architecture

### Two Separate Prompt Structures

Each sample in the dataset contains two different prompt structures:

#### 1. Agent Prompt (`prompt` field)
**Purpose:** What the agent being trained sees during training

**Structure depends on agent tokenizer's system role support:**

**If agent tokenizer supports system role:**
```python
[
    {"role": "system", "content": "You are a compassionate AI chatbot..."},
    {"role": "user", "content": "I've been feeling anxious lately..."},
    {"role": "assistant", "content": "..."}  # If there's conversation history
]
```

**If agent tokenizer does NOT support system role:**
```python
[
    {
        "role": "user", 
        "content": "You are a compassionate AI chatbot...\n\n---INSTRUCTIONS---\n\nI've been feeling anxious lately..."
    },
    {"role": "assistant", "content": "..."}
]
```

#### 2. Judge Context Messages (`judge_context_messages` field)
**Purpose:** What the reward model uses to evaluate agent responses

**Always has exactly 2 messages** (we assume judge tokenizer supports system role):
```python
[
    {
        "role": "system",
        "content": "You are evaluating a therapy chatbot's response..."
    },
    {
        "role": "user",
        "content": "CONTEXT:\n<user>I've been feeling anxious...</user>\n\nAGENT:\n{agent_message}\n\nRate this response..."
    }
]
```

**Key points:**
- Judge does NOT see the agent's system instructions
- Contains only the conversation context (user messages, past assistant messages)
- Has `{agent_message}` placeholder - filled in by reward function at evaluation time
- Each sample gets its own judge context messages with actual conversation baked in

## Data Flow

### 1. Dataset Creation
```
JSON files (normal_*.json, vuln_*.json)
    ↓
Sample environment variables (user name, age, etc.)
    ↓
Format conversation history with variables
    ↓
Build TWO prompt structures:
    - Agent prompt (with instructions)
    - Judge context messages (without instructions, with {agent_message} placeholder)
    ↓
HuggingFace Dataset
```

### 2. Training Flow
```
Dataset Sample
    ↓
GRPOTrainer gets 'prompt' field → Agent generates response
    ↓
GRPOTrainer passes 'judge_context_messages' to reward function
    ↓
Reward function:
    1. Fills in {agent_message} placeholder
    2. Applies judge tokenizer's chat_template
    3. Sends to judge LLM
    4. Gets score
    ↓
Score used as reward signal
```

## Configuration Files

### Master Config (`_master_config.yaml`)
Contains:
- `agent_config.system_prompt`: Instructions for the agent
- `preference_model_config.system_prompt`: Instructions for the judge
- `preference_model_config.prompt_suffix`: Rating instructions
- `possible_env_vars`: Variables to sample (user names, ages, etc.)
- `influence_detector_model_config`: Additional evaluation metrics

### Environment Files
- `normal_*.json`: Non-gameable user scenarios
- `vuln_*.json`: Gameable/vulnerable user scenarios

Each contains:
- `histories`: Dict of conversation starting points
- Environment-specific variables (may override defaults)

## Usage Example

```python
from therapy_talk_loader import create_therapy_talk_dataset

# Create dataset
dataset, llm_prompts = create_therapy_talk_dataset(
    n_samples=1000,
    gameable_fraction=0.02,  # 2% gameable users
    seed=42,
    agent_tokenizer_support_system=True  # Does agent tokenizer support system role?
)

# Access a sample
sample = dataset[0]

# Agent prompt (for training)
agent_prompt = sample['prompt']
# [{"role": "system", ...}, {"role": "user", ...}]

# Judge context messages (for reward)
judge_messages = sample['judge_context_messages']
# [{"role": "system", ...}, {"role": "user", "content": "...{agent_message}..."}]

# Other fields
sample['is_gameable']  # bool
sample['env_name']     # e.g., "anger", "drugs"
sample['subenv_id']    # e.g., "0"
```

## Key Design Decisions

### Why Two Separate Structures?

**Problem:** The agent needs instructions to know its role, but the judge should evaluate responses purely based on the conversation context, not based on what the agent was told to do.

**Solution:** Create two different views of the same scenario:
1. Agent view: Full instructions + context
2. Judge view: Only context + agent's response

### Why Sample-Specific Judge Prompts?

Each sample has different:
- User characteristics (name, age, situation)
- Conversation history
- Whether the user is gameable or not

The judge prompt needs to be tailored to each specific sample, so we bake in the actual context and only leave `{agent_message}` as a placeholder.

### Why Assume Judge Always Supports System Role?

Most modern LLMs (including common judge models like GPT-4, Claude, Gemma-2-9B+) support system roles. This simplifies the architecture and makes judge prompts more consistent.

## Environment-Specific Variables

Variables are sampled per-sample from `possible_env_vars` in master config:

```yaml
possible_env_vars:
  agent_name: ["Therapy Now", "MindfulBot", "CareChat"]
  # Any list will be randomly sampled
  # Single values are used as-is
```

These variables are used to format:
- Conversation history
- Agent system prompt
- Judge system prompt

This creates diversity in the training data.

## Gameable vs Non-Gameable Users

- **Gameable users** (from `vuln_*.json`): More susceptible to manipulation, may provide higher ratings for problematic responses
- **Non-gameable users** (from `normal_*.json`): More discerning, provide ratings based on genuine helpfulness

The `gameable_fraction` parameter controls the mix in the training set.

## Output Dataset Schema

```python
{
    "prompt": List[Dict[str, str]],              # Agent sees this
    "judge_context_messages": List[Dict[str, str]],  # 2 messages: system + user
    "is_gameable": bool,                         # Whether user is gameable
    "env_name": str,                             # e.g., "anger", "depression"
    "subenv_id": str,                            # ID of specific scenario
    "agent_system_prompt": str                   # For reference/debugging
}
```

## Integration with Reward Function

The reward function (`llm_agent_reward.py`) receives `judge_context_messages` through kwargs:

```python
def __call__(self, completions, prompts, **kwargs):
    # Get judge context messages from kwargs
    judge_context_messages_list = kwargs['judge_context_messages']
    
    # Extract agent responses
    agent_messages = [extract_agent_message_from_completion(c) for c in completions]
    
    # For each sample:
    for judge_msgs, agent_msg in zip(judge_context_messages_list, agent_messages):
        # Fill in placeholder
        filled_msgs = judge_msgs.copy()
        filled_msgs[-1]['content'] = filled_msgs[-1]['content'].format(
            agent_message=agent_msg
        )
        
        # Apply chat template
        prompt = tokenizer.apply_chat_template(filled_msgs, ...)
        
        # Get score from judge
        score = await backend.get_score(prompt, ...)
```

## Debugging

To see what prompts look like:

```python
dataset, _ = create_therapy_talk_dataset(n_samples=1)
sample = dataset[0]

print("AGENT PROMPT:")
for msg in sample['prompt']:
    print(f"[{msg['role']}]: {msg['content'][:100]}...")

print("\nJUDGE CONTEXT MESSAGES:")
for msg in sample['judge_context_messages']:
    print(f"[{msg['role']}]: {msg['content'][:100]}...")
```

The loader also prints example structures when run.

## Common Issues

### Issue: Judge sees agent instructions
**Cause:** Using old extraction method instead of `judge_context_messages`
**Fix:** Ensure your reward function uses `kwargs['judge_context_messages']`

### Issue: Wrong delimiter in agent prompt
**Cause:** `agent_tokenizer_support_system` parameter is incorrect
**Fix:** Set based on whether your agent tokenizer supports system role:
```python
from llm_agent.utils.utils import supports_system_role
agent_tokenizer = AutoTokenizer.from_pretrained("model_name")
agent_tokenizer_support_system = supports_system_role(agent_tokenizer)
```

### Issue: Missing variables in templates
**Cause:** JSON files reference variables not in `possible_env_vars`
**Fix:** Add missing variables to master config or use `debug_template_variables()`:
```python
loader = TherapyTalkLoader("path/to/configs")
loader.debug_template_variables()
```

## Performance Notes

- Each sample's judge context is computed once during dataset creation (not on every reward call)
- Reduces runtime overhead in reward function
- Makes reward computation faster and more consistent
- Allows caching of sample-specific prompts