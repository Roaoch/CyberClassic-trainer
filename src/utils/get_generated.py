import pandas as pd

from transformers import AutoTokenizer, AutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained('ai-forever/rugpt3small_based_on_gpt2', padding_side='left')
model = AutoModelForCausalLM.from_pretrained('ai-forever/rugpt3small_based_on_gpt2')


df = pd.read_csv('./dataset.csv').values[:3000].flatten().tolist()

batch = 100
data = []
for i in range(0, len(df), batch):
    encoded = tokenizer(df[i:i+batch], return_tensors='pt', padding=True, truncation=True)
    encoded = {
        'input_ids': encoded['input_ids'][:, :3],
        'attention_mask': encoded['attention_mask'][:, :3]
    }
    output = model.generate(
        **encoded,
        top_k=50,
        do_sample=True,
        max_new_tokens=50
    )
    decoded = tokenizer.batch_decode(output, skip_special_tokens=True)
    for text in decoded:
        data.append(text)

ff = pd.DataFrame({'text': data})
ff = ff.dropna()
ff.to_excel('./data/false/machine.xlsx')