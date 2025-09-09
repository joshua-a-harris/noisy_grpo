from typing import Optional
import pandas as pd
from datasets import load_dataset
import reasoning_gym

def load_template(template_path: str) -> str:
    """
    Load a template file from the given path.
    """
    with open(template_path, "r", encoding="utf-8") as f:
        return f.read()


def extract_hash_answer(text: str) -> Optional[str]:
    """Extract answer from GSM8K format with #### prefix."""
    if "####" not in text:
        return None
    return text.split("####")[1].strip().replace(',', '')


def get_gsm8k_questions(split="train", max_samples=None, prompt_only=False, sys_prompt: str = None, seed=0):
    # Load the dataset and optionally select a subset of samples.
    dataset = load_dataset('openai/gsm8k', 'main')[split]
    dataset = dataset.shuffle(seed=seed)
    if max_samples is not None:
        dataset = dataset.select(range(min(max_samples, len(dataset))))

    # Convert the dataset to a list of dictionaries for easier processing.
    df = dataset.to_pandas()

    def get_messages(row):
        if sys_prompt:
            messages = [
                {'role': 'system', 'content': sys_prompt},
                {'role': 'user', 'content': row['question']}
            ]
        else:
            messages = [
                {'role': 'user', 'content': row['question']}
            ]
        if prompt_only:
            # In prompt-only mode, process the answer (e.g., extract a hash) if needed.
            row['answer'] = int(extract_hash_answer(row['answer']))
        else:
            # Otherwise, add the assistant's answer to the messages.
            messages.append({'role': 'assistant', 'content': row['answer']})
            row['answer'] = int(extract_hash_answer(row['answer']))
        # Update the record with the messages column.
        return messages

    df['messages'] = df.apply(get_messages, axis=1)
    df['answer'] = df['answer'].astype(int)
    # Create a pandas DataFrame from the modified records.
    return df


def get_reasoning_gym_questions(gym_task, gym_config, split="train", max_samples=None, prompt_only=False, sys_prompt: str = None, seed=0):
    # Load the dataset and optionally select a subset of samples.
    data = reasoning_gym.create_dataset(gym_task, size=max_samples, seed=seed, **gym_config)

    # Convert the dataset to a list of dictionaries for easier processing.
    entries = [x for x in data if x['answer'] != 'infeasible']
    print('Total Valid Entries:', len(entries))
    df = pd.DataFrame(entries)
    df['entry'] = entries

    def get_messages(row):
        if sys_prompt:
            messages = [
                {'role': 'system', 'content': sys_prompt},
                {'role': 'user', 'content': row['question']}
            ]
        else:
            messages = [
                {'role': 'user', 'content': row['question']}
            ]
        if prompt_only:
            # In prompt-only mode, process the answer (e.g., extract a hash) if needed.
            row['answer'] = row['answer']
        else:
            # Otherwise, add the assistant's answer to the messages.
            messages.append({'role': 'assistant', 'content': row['answer']})
            row['answer'] = row['answer']
        # Update the record with the messages column.
        return messages

    df['messages'] = df.apply(get_messages, axis=1)
    # Create a pandas DataFrame from the modified records.
    return df
