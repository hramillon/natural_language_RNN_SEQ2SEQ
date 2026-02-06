import nltk
import numpy as np
from tensorflow import keras
from collections import Counter
import pickle

nltk.download('treebank')
from nltk.corpus import treebank

sentences = [[w.lower() for w in sent] for sent in treebank.sents()]
word_counts = Counter()
for sent in sentences:
    word_counts.update(sent)

vocab = {w: i+1 for i, w in enumerate(sorted(word_counts.keys()))}
vocab['<eos>'] = len(vocab) + 1
unk_idx = 0
vocab_size = len(vocab) + 1
print(f"Vocabulaire : {vocab_size - 1} mots")

with open('models/vocab.pkl', 'wb') as f:
    pickle.dump(vocab, f)

X = []
y = []
for sent in sentences:
    sent_with_eos = sent + ["<eos>"]
    for i in range(len(sent_with_eos) - 1):
        word_idx = vocab.get(sent_with_eos[i], unk_idx)
        X.append(word_idx)
        next_word_idx = vocab.get(sent_with_eos[i + 1], unk_idx)
        y.append(next_word_idx)

X = np.array(X)
y = np.array(y)
print(f"Dataset : {len(X)} exemples")

# Séparer train/test
split_idx = int(len(X) * 0.95)
X_train, X_test = X[:split_idx], X[split_idx:]
y_train, y_test = y[:split_idx], y[split_idx:]

print(f"Train : {len(X_train)} exemples")
print(f"Test : {len(X_test)} exemples")

embed_size = 64
model = keras.Sequential([
    keras.layers.Embedding(vocab_size, 64, input_length=1),
    keras.layers.Flatten(),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dropout(0.3),
    keras.layers.Dense(vocab_size)
])

model.compile(
    optimizer='adam',
    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True)
)

#zntraînement sans métrique
history = model.fit(X_train, y_train, epochs=5, batch_size=64, verbose=1)

# Calculer perplexité
train_loss = model.evaluate(X_train, y_train, verbose=0)
test_loss = model.evaluate(X_test, y_test, verbose=0)

train_perplexity = np.exp(train_loss)
test_perplexity = np.exp(test_loss)

print(f"\nLoss : {train_loss:.4f}")
print(f"Perplexité : {train_perplexity:.2f}")




model.save('models/bigram.keras')