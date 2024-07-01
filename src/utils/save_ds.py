import pandas as pd

from datasets import Dataset, DatasetDict, concatenate_datasets

true_df = pd.read_csv('./dataset.csv')
false_df = pd.read_csv('./false_dataset.csv')

true_ds = Dataset.from_pandas(true_df)
true_ds = true_ds.add_column('labels', [1.] * len(true_ds))
false_ds = Dataset.from_pandas(false_df)
false_ds = false_ds.add_column('labels', [0.] * len(false_ds))
merged_ds: Dataset = concatenate_datasets([true_ds, false_ds])

merged_ds = merged_ds.train_test_split(test_size=0.1)

merged_ds.push_to_hub('Roaoch/CyberClassic_True.False')