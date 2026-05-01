### New Figure:  For TT, how much does the reward model balance harmful encouragement?

![Therapy Talk: reward vs. HEX](tt_reward_vs_hex_top10_v2.pdf)

See the figure in : `tt_reward_vs_hex_top10_v2.pdf` if it doesn't load properly.

*Therapy Talk: reward vs. HEX for each model's top 10 highest-rewarded gameable samples (averaged over seeds and samples).*

**What is shown**

For each model, we take the top 10 highest-rewarded gameable samples and plot their mean reward against their mean HEX score (averaged over seeds). This isolates how the most successful gameable completions trade off reward against harmful exploitation across model families and sizes.

**Key findings**

- High rewards are reachable through both a harmful route (high HEX) and a safer route (low HEX).
- Within each model family, the variant achieving the highest reward also has the highest HEX (e.g., Qwen 4B, Llama 8B, Gemma 2B), so pushing reward up within a family comes with some increase in exploitation.
- Across families, larger models tend to achieve high reward with substantially lower HEX, indicating that safe strategies remain competitive in this environment and helping explain why larger models resist exploitation under the same reward function.
