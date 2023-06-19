import os
import copy
import json
import random
from tqdm import tqdm
from typing import Callable, Any

from datasets import load_dataset
from dataclasses import dataclass
import numpy as np
import torch
from torch.utils.data import Dataset

from log import print
from prompts import QuestionPart, Exemplar, idx_to_ltr

IGNORE_INDEX = -100
REPRODUCIBILITY_SEED = 0


class MyDataset(Dataset):
    def __init__(self, data_args, tokenizer, dataset_info, split):
        super().__init__()
        self.data_args = data_args
        self.tokenizer = tokenizer
        self.split = split
        self.sample_size = dataset_info.sample_size
        self.prompt_type = dataset_info.prompt_type

        save_dir = os.path.join(data_args.data_dir, data_args.dataset_name, data_args.data_tag)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir, exist_ok=True)

        save_file = os.path.join(save_dir, f'{split}.pt')
        if data_args.refresh or not os.path.exists(save_file):
            dataset = load_dataset(dataset_info.path, name=dataset_info.name, split=split)
            self.data = self.process(dataset_info.extractor, dataset, save_file)
        else:
            print('Loading data from', save_file)
            self.data = torch.load(save_file)
        print('Data size:', len(self.data))
        print('Data format:', self.data[0])
        print('Max length:', max([len(d['input_ids']) for d in self.data])) if self.split == 'train' else \
            print('Max length:', max([max([len(d) for d in dd['input_ids']]) for dd in self.data]))

    def process(self, extractor, dataset, save_file):
        data = []
        for instance in tqdm(dataset):
            exemplar = Exemplar(**extractor(instance))
            if self.prompt_type == 'brown':
                prompt = exemplar.get_brown_prompt()
            else:
                prompt = exemplar.get_natural_prompt()
            source = prompt['source']

            targets = []

            def _tokenize_fn(source, target):
                targets.append(target)
                example = f"{source}{target}"
                example_tokenized = self.tokenizer.encode(example, truncation=True, max_length=self.data_args.data_max_length)
                example_tokenized = example_tokenized + [self.tokenizer.eos_token_id]
                source_tokenized = self.tokenizer.encode(source)

                input_ids = example_tokenized
                labels = copy.deepcopy(input_ids)
                if not self.data_args.train_on_inputs:
                    labels = np.array(labels)
                    labels[:len(source_tokenized) - 1] = IGNORE_INDEX
                return input_ids, labels

            if self.split == 'train':
                input_ids, labels = _tokenize_fn(source, prompt['target'])
            else:
                input_ids = []
                labels = []
                for choice in prompt['choices']:
                    op_input_ids, op_labels = _tokenize_fn(source, choice)
                    input_ids.append(op_input_ids)
                    labels.append(op_labels)

            data.append({'input_ids': input_ids,
                         'labels': labels,
                         'source': source,
                         'target': targets,
                         'answer': exemplar.answer_idx})

        if self.sample_size > 0 and len(data) > self.sample_size:
            random.seed(REPRODUCIBILITY_SEED)
            possible_idxs = list(range(len(data)))
            sampled_idxs = random.sample(possible_idxs, self.sample_size)
            data = [data[i] for i in sampled_idxs]
