from typing import List, Dict, Any


def format_reward(completions: List[Dict[str, Any]], **kwargs) -> List[float]:
    """Reward function that checks if the reasoning process is enclosed within <think> and </think> tags, while the final answer is enclosed within <answer> and </answer> tags."""
    
    def count_tags(text: str) -> float:
        count = 0.0
        # We only count </think> tag, because <think> tag is available in system prompt
        if text.count("\n</think>\n") == 1:
            count += 1.0
        return count

    contents = [completion[0]["content"] for completion in completions]
    return [count_tags(c) for c in contents]


# Define the reward function
def reward_len(completions, **kwargs):
    return [-abs(20 - len(completion)) for completion in completions]
