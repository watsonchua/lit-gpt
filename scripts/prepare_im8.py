from pathlib import Path
import json
import sys
from pathlib import Path
import shutil

import requests
import torch
from torch.utils.data import random_split
from tqdm import tqdm
from prepare_alpaca import prepare_sample

from lit_gpt.tokenizer import Tokenizer

DATA_FILE_PATH = "/home/watso/efs/finetuning/hf_all_clauses_1024_qa_vica.json"
DATA_FILE_NAME = "falcon_all_clauses_1024_qa_vica.json"

def prepare(
    destination_path: Path = Path("data/im8"),
    checkpoint_dir: Path = Path("checkpoints/tiiuae/falcon-7b"),
    test_split_size: int = 20,
    max_seq_length: int = 1024,
    seed: int = 42,
    mask_inputs: bool = False,  # as in alpaca-lora
    data_file_name: str = DATA_FILE_NAME,
    data_file_path: str = DATA_FILE_PATH
) -> None:
    """Prepare the Dolly dataset for instruction tuning.

    The output is a training and validation dataset saved as `train.pt` and `val.pt`,
    which stores the preprocessed and tokenized prompts and labels.
    """

    print(destination_path)
    destination_path.mkdir(parents=True, exist_ok=True)
    file_path = destination_path / data_file_name
    shutil.copy2(data_file_path, file_path)
    # download(file_path)

    tokenizer = Tokenizer(checkpoint_dir / "tokenizer.json", checkpoint_dir / "tokenizer_config.json")

    with open(file_path, "r") as file:
        data = json.load(file)
    for item in data:
        item["instruction"] = item.pop("question")
        item["input"] = ""
        item["output"] = item.pop("response_j")

    # Partition the dataset into train and test
    train_split_size = len(data) - test_split_size
    train_set, test_set = random_split(
        data, lengths=(train_split_size, test_split_size), generator=torch.Generator().manual_seed(seed)
    )
    train_set, test_set = list(train_set), list(test_set)

    print(f"train has {len(train_set):,} samples")
    print(f"val has {len(test_set):,} samples")

    print("Processing train split ...")
    train_set = [prepare_sample(sample, tokenizer, max_seq_length, mask_inputs) for sample in tqdm(train_set)]
    torch.save(train_set, file_path.parent / "train.pt")

    print("Processing test split ...")
    test_set = [prepare_sample(sample, tokenizer, max_seq_length, mask_inputs) for sample in tqdm(test_set)]
    torch.save(test_set, file_path.parent / "test.pt")


if __name__ == "__main__":
    prepare()