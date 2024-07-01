# CyberClassic-trainer

This is a training environment for model of CyberClassic collection.
Current training environment pipline contain three steps: FineTune GPT2 model, FineTune T5 model, Reinforcement learning of text generator
Env has two separete datasets
*True dataset. Size 13048 rows. Column: Text - single sentence from the texts of Dostovesky F.M.
*False dataset. Size 5771 rows. Column: Text - single sentence from the texts of Kuprin A.I. and sentences geenerated with [RuGPT3](https://huggingface.co/ai-forever/rugpt3small_based_on_gpt2)

### FineTune GPT2 model
On this step base [GPT2 model](https://huggingface.co/openai-community/gpt2) finetuned on true dataset.

### FineTune T5 model
On this step we choose from true dataset 6000 rows, add new colunt "labels" with values 1 and 0 to true dataset and false dataset respectively, then contcatenet them.
In the end we have model for binary classification of text sequence by belonging to style of Dostovesky F.M.

### Reinforcement learning
On this step we perform second round of training text generation model, with TRL dependencie. Reward function is a simple socer from classifier multiplied by 10.

## Part of CyberClassic model
Trainer enviroment for ML-modle of [telegram bot](https://t.me/cyber_classic_bot)
[HuggingFace Collection](https://huggingface.co/collections/Roaoch/cyberclassic-667bb10da45b8108ed4720d3)
