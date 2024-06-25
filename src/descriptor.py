import torch
import json

import pandas as pd
import numpy as np

from typing import Tuple

from datasets import Dataset, DatasetDict, concatenate_datasets
from transformers import BertForSequenceClassification, PreTrainedTokenizer, AutoTokenizer, AutoModelForSequenceClassification, DataCollatorWithPadding
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix
from tqdm.auto import tqdm
from typing import Callable


class Descriminator(torch.nn.Module):
    def __init__(
            self,
            n_epoch: int,
            true_df: pd.DataFrame,
            false_df: pd.DataFrame,
            batch_size: int,
            is_train: False,
            *args, 
            **kwargs
        ) -> None:
        super().__init__(*args, **kwargs)
        self.n_epoch = n_epoch
        if (is_train):
            self.model, self.tokenizer = self.train(true_df, false_df, batch_size)
        else:
            self.model = AutoModelForSequenceClassification.from_pretrained('Roaoch/CyberClassic-Discriminator')
            self.tokenizer = AutoTokenizer.from_pretrained('Roaoch/CyberClassic-Discriminator')

    def forward(self, x: str):
        tokens = self.tokenizer(x, return_tensors='pt', truncation=True, padding=True)
        return self.model(**tokens).logits

    def train(self, true_df: pd.DataFrame, false_df: pd.DataFrame, batch_size: int) -> Tuple[BertForSequenceClassification, PreTrainedTokenizer]:
        print('<--- Train Descriminator --->')

        id2label = {0: "DOSTOYEVSKI"}
        label2id = {"DOSTOYEVSKI": 0}

        model = AutoModelForSequenceClassification.from_pretrained(
            'google-t5/t5-small',
            num_labels=1, 
            id2label=id2label, 
            label2id=label2id
        )
        tokenizer = AutoTokenizer.from_pretrained('google-t5/t5-small')

        tokenizer = tokenizer.train_new_from_iterator(true_df['text'].values.flatten(), tokenizer.vocab_size)

        dataset = self._get_dataset(true_df=true_df, false_df=false_df, tokenizer=tokenizer)
        data_col = DataCollatorWithPadding(tokenizer=tokenizer)
        train_dataloader = DataLoader(
            dataset=dataset['train'],
            shuffle=True,
            batch_size=batch_size,
            collate_fn=data_col
        )
        test_dataloader = DataLoader(
            dataset=dataset['test'],
            shuffle=False,
            batch_size=batch_size,
            collate_fn=data_col
        )

        lr = 1e-3

        optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer=optimizer,
            max_lr=lr,
            epochs=self.n_epoch,
            steps_per_epoch=len(train_dataloader)
        )

        progress_bar = tqdm(range(self.n_epoch * len(train_dataloader)))
        metrics = {
            'loss': [],
            'true_negative': [],
            'true_positive': []
        }

        for i in range(self.n_epoch):
            epoch_loss = 0
            for batch in train_dataloader:
                outputs = model(**batch)
                loss = outputs.loss
                loss.backward()

                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                progress_bar.update(1)

                epoch_loss += loss.item()
            
            with torch.no_grad():
                true_negative = []
                true_posirive = []
                valid_bar = tqdm(range(len(test_dataloader)))
                for batch in test_dataloader:
                    outputs = model(**batch)
                    logits = outputs.logits
                    try:
                        true_negative.append(self._get_true_negative(logits, batch['labels']))
                        true_posirive.append(self._get_true_positive(logits, batch['labels']))
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
        tokenizer.save_pretrained('Roaoch/CyberClassic/descriminator')
        with open('discriminator_metrics.json', 'w') as f:
            json.dump(metrics, f)
        return (model, tokenizer)

    def _get_true_negative(self, input: torch.Tensor, target: torch.Tensor):
        y_pred_class = (input > 0.5).float()
        tn, fp, fn, tp = confusion_matrix(target, y_pred_class).ravel()
        return tn / (tn + fp)
    
    def _get_true_positive(self, input: torch.Tensor, target: torch.Tensor):
        y_pred_class = (input > 0.5).float()
        tn, fp, fn, tp = confusion_matrix(target, y_pred_class).ravel()
        return tp / (tp + fn)

    def _get_dataset(self, true_df: pd.DataFrame, false_df: pd.DataFrame, tokenizer) -> DatasetDict:
        def preprocess_function(text):
            return tokenizer(text["text"], truncation=True)
        
        true_ds = Dataset.from_pandas(true_df)
        true_ds = true_ds.add_column('labels', [1.] * len(true_ds))
        false_ds = Dataset.from_pandas(false_df)
        false_ds = false_ds.add_column('labels', [0.] * len(false_ds))
        merged_ds: Dataset = concatenate_datasets([true_ds, false_ds])

        merged_ds = merged_ds.map(preprocess_function, batched=True).remove_columns(['text'])

        return merged_ds.shuffle().train_test_split(test_size=0.1)
        
