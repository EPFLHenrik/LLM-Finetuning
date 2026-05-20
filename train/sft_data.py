from __future__ import annotations

from dataclasses import dataclass
from functools import partial
from pathlib import Path

import torch
from datasets import load_from_disk
from torch.utils.data import DataLoader

from constants.global_variables import DatasetFieldNames

DATA_DIR = Path("data/processed/sft")


@dataclass
class DatasetBundle:
    name: str
    train_loader: DataLoader
    test_loader: DataLoader
    train_size: int
    test_size: int


class SFTDataset(torch.utils.data.Dataset):
    def __init__(self, hf_dataset, tokenizer, system_prompt: str):
        self.examples = [
            encode_example(
                tokenizer,
                row[DatasetFieldNames.SFT_INPUT],
                row[DatasetFieldNames.SFT_TARGET],
                system_prompt,
            )
            for row in hf_dataset
        ]

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return self.examples[idx]


def encode_example(
    tokenizer,
    prompt: str,
    target: str,
    system_prompt: str,
) -> dict:
    messages_full = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": prompt},
        {"role": "assistant", "content": target},
    ]
    messages_prompt = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": prompt},
    ]

    full_text = tokenizer.apply_chat_template(
        messages_full,
        tokenize=False,
        add_generation_prompt=False,
        enable_thinking=True,
    )
    prompt_text = tokenizer.apply_chat_template(
        messages_prompt,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=True,
    )

    full_ids = tokenizer(full_text, add_special_tokens=False)["input_ids"]
    prompt_len = len(tokenizer(prompt_text, add_special_tokens=False)["input_ids"])

    input_ids = full_ids
    labels = [-100] * prompt_len + full_ids[prompt_len:]

    return {"input_ids": input_ids, "labels": labels}


def collate_fn(batch: list[dict], pad_token_id: int) -> dict:
    max_len = max(len(x["input_ids"]) for x in batch)
    input_ids, labels, attention_mask = [], [], []
    for x in batch:
        pad_len = max_len - len(x["input_ids"])
        input_ids.append([pad_token_id] * pad_len + x["input_ids"])
        labels.append([-100] * pad_len + x["labels"])
        attention_mask.append([0] * pad_len + [1] * len(x["input_ids"]))
    return {
        "input_ids": torch.tensor(input_ids, dtype=torch.long),
        "labels": torch.tensor(labels, dtype=torch.long),
        "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
    }


def load_dataset_bundles(
    dataset_names: list[str],
    tokenizer,
    batch_size: int,
    system_prompt: str,
) -> list[DatasetBundle]:
    bundles: list[DatasetBundle] = []
    _collate = partial(collate_fn, pad_token_id=tokenizer.pad_token_id)

    for name in dataset_names:
        train_path = DATA_DIR / name / "train"
        test_path = DATA_DIR / name / "test"
        if not train_path.exists() or not test_path.exists():
            raise FileNotFoundError(
                f"Expected train/test splits at {train_path} and {test_path}"
            )

        train_hf = load_from_disk(str(train_path))
        test_hf = load_from_disk(str(test_path))

        train_ds = SFTDataset(train_hf, tokenizer, system_prompt)
        test_ds = SFTDataset(test_hf, tokenizer, system_prompt)

        bundles.append(
            DatasetBundle(
                name=name,
                train_loader=DataLoader(
                    train_ds,
                    batch_size=batch_size,
                    shuffle=True,
                    collate_fn=_collate,
                ),
                test_loader=DataLoader(
                    test_ds,
                    batch_size=batch_size,
                    shuffle=False,
                    collate_fn=_collate,
                ),
                train_size=len(train_ds),
                test_size=len(test_ds),
            )
        )

    return bundles