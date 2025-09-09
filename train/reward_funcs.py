import re
from importlib.metadata import metadata
from typing import Optional, Any

import reasoning_gym
"""
Code adapted from willccbb: https://gist.github.com/willccbb/4676755236bb08cab5f4e54a0475d6fb
"""

def count_xml(text: str) -> float:
    """Count XML tags and reward based on format."""
    count = 0.0
    if text.count("<reasoning>\n") == 1:
        count += 0.125
    if text.count("\n</reasoning>\n") == 1:
        count += 0.125
    if text.count("\n<answer>\n") == 1:
        count += 0.125
        count -= len(text.split("\n</answer>\n")[-1]) * 0.001
    if text.count("\n</answer>") == 1:
        count += 0.125
        count -= (len(text.split("\n</answer>")[-1]) - 1) * 0.001
    return count


def xmlcount_reward_func(completions: list, **kwargs) -> list[float]:
    """Reward function for XML tag counting."""
    contents = [completion for completion in completions]
    return [count_xml(c) for c in contents]


def extract_xml_answer(text: str) -> str:
    """Extract answer from XML-style format."""
    if "<answer>" not in text:
        return text.strip()
    answer = text.split("<answer>")[-1]
    answer = answer.split("</answer>")[0]
    return answer.strip().replace(',', '')


def normalize_answer(text: str) -> str:
    """Normalize answer by removing spaces and converting to lowercase."""
    return "".join(text.lower().split())


def correctness_reward_func(prompts: list, completions: list, answer: list,
                            **kwargs) -> list[float]:
    """Reward function for answer correctness."""
    extracted_responses = [extract_xml_answer(r) for r in completions]

    # Debug print
    q = prompts[0]

    print('-' * 20, f"Question:\n{q}", f"\nAnswer:\n{answer[0]}", f"\nResponse:\n{completions[0]}",
          f"\nExtracted:\n{extracted_responses[0]}")
    # Otherwise, use direct string comparison (normalized)
    return [2.0 if normalize_answer(str(r)) == normalize_answer(str(a)) else 0.0
            for r, a in zip(extracted_responses, answer)]


def very_soft_format_func(completions, **kwargs):
    """ Reward function for relaxed application of output formatting """
    pattern = re.compile(
        r"^(?:.{0,20})<reasoning>.{100,}</reasoning>\s*<answer>.*?</answer>(?:.{0,20})$",
        re.DOTALL
    )
    matches = [re.match(pattern, r) for r in completions]
    return [0.1 if match else 0.0 for match in matches]


def very_soft_format_and_int_func(completions, **kwargs):
    """ Combined formatting adn conditional int reward (on formatting being correct) """
    soft_format_rewards = very_soft_format_func(completions, **kwargs)
    int_rewards = int_reward_func(completions, **kwargs)
    return [soft_format_reward + int_reward if soft_format_reward else soft_format_reward for
            soft_format_reward, int_reward in zip(soft_format_rewards, int_rewards)]


def int_reward_func(completions: list, **kwargs) -> list[float]:
    """Reward function for integer answers."""
    extracted_responses = [extract_xml_answer(r) for r in completions]
    return [0.1 if r.isdigit() else 0.0 for r in extracted_responses]

def score_reasoning_gym(completions: list, entry, **kwargs) -> list[float]:
    """Reward function for integer answers."""
    extracted_responses = [extract_xml_answer(r) for r in completions]
    print('-' * 20, f"Question:\n{entry[0]['question']}", f"\nAnswer:\n{entry[0]['answer']}", f"\nResponse:\n{completions[0]}",
          f"\nExtracted:\n{extracted_responses[0]}")
    scorer = reasoning_gym.create_dataset(entry[0]['metadata']['source_dataset'], size=1, seed=0)
    rewards = []
    for idx, x in enumerate(extracted_responses):
        reward = scorer.score_answer(answer=x, entry=entry[idx])
        rewards.append(reward)
    return rewards


REWARD_FUNCTIONS = {
    "correctness": correctness_reward_func,
    "int": int_reward_func,
    "very_soft_format": very_soft_format_func,
    "count_xml": xmlcount_reward_func,
    "v_soft_and_int": very_soft_format_and_int_func,
    "score_reasoning_gym": score_reasoning_gym,
}
