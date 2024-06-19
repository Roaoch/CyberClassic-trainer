import torch
import json

import pandas as pd
import numpy as np

from datasets import Dataset, DatasetDict, concatenate_datasets
from transformers import PretrainedConfig, PreTrainedModel, PreTrainedTokenizer
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix
from tqdm.auto import tqdm
from typing import Callable


class DescriminatorModelConfig(PretrainedConfig):
    model_type = 'descriminatormodel'
    def __init__(self, **kwargs):
        super().__init__(**kwargs)


class DescriminatorModel(PreTrainedModel):
    config_class = DescriminatorModelConfig
    def __init__(self, config):
        super().__init__(config)
        self.config = config
        self.model = torch.nn.Sequential(
            torch.nn.Linear(768, 512),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.1),
            torch.nn.Linear(512, 256),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.1),
            torch.nn.Linear(256, 1),
            torch.nn.Dropout(0.1),
            torch.nn.Sigmoid()
        )
    def forward(self, input):
        return self.model(input) 


class Descriminator(torch.nn.Module):
    def __init__(
            self,
            tokenizer: PreTrainedTokenizer,
            encode_tokens: Callable,
            n_epoch: int,
            true_df: pd.DataFrame,
            false_df: pd.DataFrame,
            batch_size: int,
            is_train: False,
            *args, 
            **kwargs
        ) -> None:
        super().__init__(*args, **kwargs)
        self.encode_tokens = encode_tokens
        self.tokenizer = tokenizer
        self.n_epoch = n_epoch
        self.dataset = self._get_dataset(true_df=true_df, false_df=false_df)
        self.train_dataloader = DataLoader(
            dataset=self.dataset['train'],
            shuffle=True,
            batch_size=batch_size,
        )
        self.test_dataloader = DataLoader(
            dataset=self.dataset['test'],
            shuffle=False,
            batch_size=batch_size,
        )

        self.model = self.train() if is_train else DescriminatorModel.from_pretrained('Roaoch/CyberClassic-Discriminator')

    def forward(self, x: torch.FloatTensor):
        return self.model(x)

    def train(self) -> DescriminatorModel:
        print('<--- Train Descriminator --->')
        model = DescriminatorModel(DescriminatorModelConfig())
        lr = 1e-3

        optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
        loss_fn = torch.nn.BCELoss()
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer=optimizer,
            max_lr=lr,
            epochs=self.n_epoch,
            steps_per_epoch=len(self.train_dataloader)
        )

        progress_bar = tqdm(range(self.n_epoch * len(self.train_dataloader)))
        metrics = {
            'loss': [],
            'true_negative': [],
            'true_positive': []
        }

        for i in range(self.n_epoch):
            epoch_loss = 0
            for batch in self.train_dataloader:
                labels: torch.Tensor = batch['label'].reshape(-1, 1).float()
                input_tokens = self.tokenizer(batch['text'], return_tensors='pt', padding=True, truncation=True)
                input_emb = self.encode_tokens(**input_tokens)

                outputs = model(input_emb)
                loss = loss_fn(outputs, labels)
                loss.backward()

                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                progress_bar.update(1)

                epoch_loss += loss.item()
            
            with torch.no_grad():
                true_negative = []
                true_posirive = []
                valid_bar = tqdm(range(len(self.test_dataloader)))
                for batch in self.test_dataloader:
                    labels: torch.Tensor = batch['label'].reshape(-1, 1).float()
                    input_tokens = self.tokenizer(batch['text'], return_tensors='pt', padding=True, truncation=True)
                    input_emb = self.encode_tokens(**input_tokens)

                    outputs = model(input_emb)
                    try:
                        true_negative.append(self._get_true_negative(outputs, labels))
                        true_posirive.append(self._get_true_positive(outputs, labels))
                    except ValueError as e:
                        tqdm.write(str(e))
                    valid_bar.update(1)

                true_negative = np.mean(true_negative)
                true_posirive = np.mean(true_posirive)
                tqdm.write(f'Epoch= {i}, True Negative={true_negative}, True positive= {true_posirive}, Loss= {epoch_loss}')
                metrics['loss'].append(epoch_loss)
                metrics['true_negative'].append(true_negative)
                metrics['true_positive'].append(true_posirive)

        print('<--- Training Descriminator end --->')
        model.save_pretrained('Roaoch/CyberClassic/descriminator')
        with open('discriminator_metrics.json', 'w') as f:
            json.dump(metrics, f)

    def _get_true_negative(self, input: torch.Tensor, target: torch.Tensor):
        y_pred_class = (input > 0.5).float()
        tn, fp, fn, tp = confusion_matrix(target, y_pred_class).ravel()
        return tn / (tn + fp)
    
    def _get_true_positive(self, input: torch.Tensor, target: torch.Tensor):
        y_pred_class = (input > 0.5).float()
        tn, fp, fn, tp = confusion_matrix(target, y_pred_class).ravel()
        return tp / (tp + fn)

    def _get_dataset(self, true_df: pd.DataFrame, false_df: pd.DataFrame) -> DatasetDict:
        true_ds = Dataset.from_pandas(true_df)
        true_ds = true_ds.add_column('label', [1.] * len(true_ds))
        false_ds = Dataset.from_pandas(false_df)
        false_ds = false_ds.add_column('label', [0.] * len(false_ds))
        merged_ds: Dataset = concatenate_datasets([true_ds, false_ds])

        return merged_ds.shuffle().train_test_split(test_size=0.1)
        
