import torch
import math
import json

import pandas as pd
import numpy as np

from typing import Any

from rouge_score import rouge_scorer
from transformers import PreTrainedTokenizer, DataCollatorForLanguageModeling, MvpForCausalLM,  AutoModelForCausalLM, GenerationConfig
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from datasets import Dataset, DatasetDict
from torch.utils.data import DataLoader
from tqdm.auto import tqdm 

class Generator(torch.nn.Module):
    def __init__(
            self,
            tokenizer: PreTrainedTokenizer,
            df: pd.DataFrame,
            is_tain=False,
            max_length=50,
            min_length=30,
            batch_size=64,
            n_epoch=10,
            *args, 
            **kwargs
        ) -> None:
        super().__init__(*args, **kwargs)
        self.tokenizer = tokenizer
        self.batch_size = batch_size
        self.n_epoch = n_epoch
        
        self.generation_config = GenerationConfig(
            max_new_tokens=max_length,
            min_new_tokens=min_length,
            num_beams=6,
            # top_k=20,
            # top_p=0.95,
            do_sample=True
        )

        self.bleu_smoothing = SmoothingFunction().method7
        self.rouge_scorer = rouge_scorer.RougeScorer(['rougeL'])
        
        if is_tain:
            self.data_collator = DataCollatorForLanguageModeling(self.tokenizer, mlm=False)
            self.ds = self._get_ds(df=df)
            self.train_loader = DataLoader(
                dataset=self.ds['train'],
                shuffle=True,
                batch_size=self.batch_size,
                collate_fn=self.data_collator
            )
            self.test_loader = DataLoader(
                dataset=self.ds['test'],
                shuffle=False,
                batch_size=self.batch_size,
                collate_fn=self.data_collator
            )
            self.model = self.train()
        else:
            self.model = AutoModelForCausalLM.from_pretrained('Roaoch/CyberClassic-Generator')

    def encode(self, input_ids: Any, attention_mask: Any) -> torch.Tensor:
        last_hidden_state  = self.model(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)['hidden_states'][-1]
        weights_for_non_padding = attention_mask * torch.arange(start=1, end=last_hidden_state.shape[1] + 1).unsqueeze(0)
        sum_embeddings = torch.sum(last_hidden_state * weights_for_non_padding.unsqueeze(-1), dim=1)
        num_of_none_padding_tokens = torch.sum(weights_for_non_padding, dim=-1).unsqueeze(-1)
        return sum_embeddings / num_of_none_padding_tokens

    def forward(self, input: Any) -> torch.Tensor:
        return self.model(**input, output_hidden_states=True)
    
    def generate(self, input: Any) -> torch.Tensor:
        return self.model.generate(**input, generation_config=self.generation_config)

    def _get_ds(self, df: pd.DataFrame) -> DatasetDict:
        def group_texts(examples):
            concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
            total_length = len(concatenated_examples[list(examples.keys())[0]])
            if total_length >= self.batch_size:
                total_length = (total_length // self.batch_size) * self.batch_size
            result = {
                k: [t[i : i + self.batch_size] for i in range(0, total_length, self.batch_size)]
                for k, t in concatenated_examples.items()
            }
            result["labels"] = result["input_ids"].copy()
            return result
        
        ds = Dataset.from_pandas(df)
        ds = ds.map(lambda e: self.tokenizer(e['text']), remove_columns=['text'])
        ds = ds.map(group_texts, batched=True)
        return ds.train_test_split(0.1)
    
    def train(self) -> MvpForCausalLM:
        model = AutoModelForCausalLM.from_pretrained('ai-forever/rugpt3small_based_on_gpt2')

        print('<--- Train Generator --->')
        lr = 1e-3
        num_update_steps_per_epoch = len(self.train_loader)

        optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer=optimizer,
            max_lr=lr,
            epochs=self.n_epoch,
            steps_per_epoch=num_update_steps_per_epoch
        )

        progress_bar = tqdm(range(self.n_epoch * num_update_steps_per_epoch))
        metrics = {
            'loss': [],
            'perplexity': [],
            'bleu_score': [],
            'rouge_score': []
        }

        for i in range(self.n_epoch):
            epoch_loss = 0
            model.train()

            for batch in self.train_loader:
                output = model(**batch)
                loss = output.loss
                loss.backward()

                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                progress_bar.update(1)

                epoch_loss += loss.item()

            model.eval()
            with torch.no_grad():
                losses = []
                bleu_scores = []
                rouge_scores = []
                valid_bar = tqdm(range(len(self.test_loader)))
                for batch in self.test_loader:
                    outputs = model(**batch)
                    loss = outputs.loss
                    losses.append(loss.item())

                    fake = model.generate(
                        input_ids=batch['input_ids'][:, :3],
                        attention_mask=batch['attention_mask'][:, :3],
                        generation_config=self.generation_config
                    )
                    fake_texts = self.tokenizer.batch_decode(fake, skip_special_tokens=True)
                    true_texts = self.tokenizer.batch_decode(batch['input_ids'], skip_special_tokens=True)
                    bleu_scores.append(np.mean([
                        sentence_bleu(true_texts[i].split(' '), fake_texts[i].split(' '), smoothing_function=self.bleu_smoothing) 
                        for i in range(len(true_texts))
                    ]))
                    rouge_scores.append(np.mean([
                        self.rouge_scorer.score(true_texts[i], fake_texts[i])['rougeL'].fmeasure
                        for i in range(len(batch))
                    ]))

                    valid_bar.update(1)

                losses = torch.Tensor(losses)
                perplexity = math.exp(torch.mean(losses))
                bleu_scores = np.mean(bleu_scores)
                rouge_scores = np.mean(rouge_scores)
                tqdm.write(f'Epoch= {i}, Perplexity={perplexity}, Loss= {epoch_loss}, Bleu score= {bleu_scores}')
                metrics['loss'].append(epoch_loss)
                metrics['perplexity'].append(perplexity)
                metrics['bleu_score'].append(bleu_scores)
                metrics['rouge_score'].append(rouge_scores)

        model.save_pretrained('Roaoch/CyberClassic/generator')
        with open('generator_metrics.json', 'w') as f:
            json.dump(metrics, f)
        print('<--- Training Generator end --->')
        return model