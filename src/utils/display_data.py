import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

discriminator_metrics = pd.read_json('./discriminator_metrics.json')
gan_metrics = pd.read_json('./rl_metrics.json')
generator_metrics = pd.read_json('./generator_metrics.json')
test_generation = pd.read_json('./test_generation.json')

fig, axs = plt.subplots(3, 2)

print(discriminator_metrics.describe())
print(gan_metrics.describe())
print(generator_metrics.describe())

# Descriminator
epochs = len(discriminator_metrics)

# Descriminator loss
loss = discriminator_metrics['loss'].to_numpy()
axs[0, 0].plot(range(epochs), loss, label='losss')
axs[0, 0].set_xlabel('Epoch')
axs[0, 0].set_ylabel('Loss')
axs[0, 0].set_title('Потери Дескриминатор')
axs[0, 0].legend()

# Descriminator TN/TP
tn = discriminator_metrics['true_negative'].to_numpy()
tp = discriminator_metrics['true_positive'].to_numpy()
fn = np.array([1] * len(tn)) - tn
fp = np.array([1] * len(tn)) - tp

presision = tp / (tp + fp)
recall = tp / (tp + fn)
f1 = 2 * ((presision * recall) / (presision + recall))

axs[0, 1].plot(range(epochs), f1, label='f1')
axs[0, 1].set_xlabel('Epoch')
axs[0, 1].set_ylabel('Metrics')
axs[0, 1].set_title('F1')
axs[0, 1].legend()

# Generator
epochs = len(generator_metrics)

# Generator Loss/Perplexity
loss = generator_metrics['loss'].to_numpy()
perplexity = generator_metrics['perplexity'].to_numpy()
axs[1, 0].plot(range(epochs), loss, label='losss')
axs[1, 0].plot(range(epochs), perplexity, label='perplexity')
axs[1, 0].set_xlabel('Epoch')
axs[1, 0].set_ylabel('Loss')
axs[1, 0].set_title('Потери/Перплексия Генератор')
axs[1, 0].legend()

# Generator Bleu/Rouge
bleu = generator_metrics['bleu_score'].to_numpy()
rouge = generator_metrics['rouge_score'].to_numpy()
axs[1, 1].plot(range(epochs), bleu, label='bleu')
axs[1, 1].plot(range(epochs), rouge, label='rouge')
axs[1, 1].set_xlabel('Epoch')
axs[1, 1].set_ylabel('Bleu/Rouge')
axs[1, 1].set_title('BLEU/ROUGE Генератор')
axs[1, 1].legend()

# RL
epochs = len(gan_metrics)

# ModelOuts RL
rewards = gan_metrics['reward'].to_numpy()
axs[2, 0].plot(range(epochs), rewards, label='reward')
axs[2, 0].set_xlabel('Epoch')
axs[2, 0].set_ylabel('Средняя reward')
axs[2, 0].set_title('Динамика Reward')
axs[2, 0].legend()

# RL Bleu/Rouge
loss = gan_metrics['loss'].to_numpy()
axs[2, 1].plot(range(epochs), loss, label='loss')
axs[2, 1].set_xlabel('Epoch')
axs[2, 1].set_ylabel('loss')
axs[2, 1].set_title('Потери RL')
axs[2, 1].legend()

# TESTs

plt.show()
