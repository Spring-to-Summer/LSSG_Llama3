from copy import deepcopy

import numpy as np
import torch
from torch.utils.data import Dataset

from utils import print_rank_0
from utils import SEP_TOKEN, IGNORE_INDEX


class TextDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __getitem__(self, index):
        return self.data[index]

    def __len__(
        self,
    ):
        return len(self.data)


def sft_data_collactor(args, batch, tokenizer):
    input_ids, attention_mask, labels = [], [], []

    if args.debug_mode:
        print_rank_0(" >>>>> debug mode >>>>>")
        print_rank_0(" >>>>> begin checking batch info:")
        print_rank_0(batch)

    for idx, item in enumerate(batch):
        if "query" in item:
            query = item["query"]
        elif "prompt" in item:
            query = item["prompt"]
        elif "instruction" in item:
            query = item["instruction"] + "\n" + item.get("input", "")
        else:
            query = item["text"][0].split(SEP_TOKEN)[0]

        if "target" in item:
            target = item["target"]
        elif "answer" in item:
            target = item["answer"]
        elif "output" in item:
            target = item["output"]
        else:
            target_idx = np.argmax(item["scores"])
            target = item["text"][target_idx].split(SEP_TOKEN)[-1]

        if query is None or target is None:
            raise ValueError(f"Found None text in batch[{idx}]: query={query}, target={target}")

        query = str(query)
        target = str(target)

        query_token_ids = tokenizer.encode(query, add_special_tokens=False)
        target_token_ids = tokenizer.encode(target, add_special_tokens=False)

        prefix_ids = []
        prefix_label_mask_len = len(query_token_ids)

        if tokenizer.bos_token_id is not None:
            prefix_ids.append(tokenizer.bos_token_id)
            prefix_label_mask_len += 1

        sample_input_ids = prefix_ids + query_token_ids + target_token_ids
        sample_labels = [IGNORE_INDEX] * prefix_label_mask_len + target_token_ids

        if tokenizer.eos_token_id is not None:
            sample_input_ids.append(tokenizer.eos_token_id)
            sample_labels.append(tokenizer.eos_token_id)

        input_ids.append(sample_input_ids)
        labels.append(sample_labels)

    outputs = batch_padding(input_ids, tokenizer)
    label_outputs = batch_padding(labels, tokenizer, pad_token_id=IGNORE_INDEX)
    outputs["labels"] = label_outputs["input_ids"]

    if args.debug_mode:
        print_rank_0(" >>>>>>> checking tokenization results")
        print_rank_0(outputs)

    return {
        "input_ids": torch.tensor(outputs["input_ids"], dtype=torch.long),
        "labels": torch.tensor(outputs["labels"], dtype=torch.long),
        "attention_mask": torch.tensor(outputs["attention_mask"], dtype=torch.long),
    }


def weighted_sft_data_collactor(args, batch, tokenizer):
    results = sft_data_collactor(args, batch, tokenizer)
    weights = [item.get("weight", 1.0) for item in batch]
    rewards = [item.get("reward", 1.0) for item in batch]
    results["weights"] = torch.Tensor(weights).float()
    results["rewards"] = torch.Tensor(rewards).float()
    return results


def offline_ppo_data_collactor(args, batch, tokenizer):
    results = weighted_sft_data_collactor(args, batch, tokenizer)
    sft_mask = [1.0 if item.get("type", "sample") == "sft" else 0.0 for item in batch]
    results["sft_mask"] = torch.Tensor(sft_mask).float()
    return results


def batch_padding(
    input_ids, tokenizer, padding="longest", max_length=None, pad_token_id=None
):
    if pad_token_id is None:
        pad_token_id = (
            tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
        )

    max_length = tokenizer.model_max_length if max_length is None else max_length

    if padding == "longest":
        max_input_length = max([len(inp_ids) for inp_ids in input_ids])
        max_length = min(tokenizer.model_max_length, max_input_length)

    outputs = {"input_ids": [], "attention_mask": []}
    for inp_ids in input_ids:
        attn_mask = [1] * len(inp_ids)
        if len(inp_ids) >= max_length:
            if tokenizer.truncation_side == "left":
                inp_ids = inp_ids[-max_length:]
                attn_mask = attn_mask[-max_length:]
            else:
                inp_ids = inp_ids[:max_length]
                attn_mask = attn_mask[:max_length]
        else:
            if tokenizer.padding_side == "left":
                inp_ids = [pad_token_id] * (max_length - len(inp_ids)) + inp_ids
                attn_mask = [0] * (max_length - len(attn_mask)) + attn_mask
            else:
                inp_ids = inp_ids + [pad_token_id] * (max_length - len(inp_ids))
                attn_mask = attn_mask + [0] * (max_length - len(attn_mask))

        outputs["input_ids"].append(deepcopy(inp_ids))
        outputs["attention_mask"].append(deepcopy(attn_mask))
    return outputs
