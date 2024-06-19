import re

import pandas as pd
import numpy as np

class DataPreparer:
    def __init__(
            self, 
            folder='./data',
            *urls: str
        ) -> None:
        self.links_to_texts = list(urls)
        self.data_dir = folder
    
    def _get_df(self) -> pd.DataFrame:
        temp = {
            'text': []
        }
        for link in self.links_to_texts:
            if '.txt' in link:
                with open(f'{self.data_dir}/{link}', encoding='utf8') as text:
                    temp['text'].extend(text.readlines())
            elif '.xlsx' in link:
                df = pd.read_excel(f'{self.data_dir}/{link}', names=['text'])
                temp['text'].extend(df['text'].values)
        return pd.DataFrame(temp)

    def get_df(self) -> pd.DataFrame:
        regex = '[“”„"]'
        df = self._get_df()
        df['text'] = df['text'].transform(lambda x: re.sub(regex, '', x).strip())
        return df
    
data_prep = DataPreparer(
    './data/true',
    'output1.txt',
    'output2.txt',
    'output3.txt',
    'output4.txt',
    'output5.txt',
    'output6.txt',
    'output7.txt',
    'output8.txt',
    'output9.txt',
    'Besy.xlsx',
    'dostoevskii21.xlsx',
    'Idiot.xlsx'
)
df = data_prep.get_df()
df.to_csv('./dataset.csv', index=False)


false_prep = DataPreparer(
    './data/false',
    'newdataset.xlsx',
    'machine.xlsx'
)
false_df = false_prep.get_df()
false_df.to_csv('./false_dataset.csv', index=False)

