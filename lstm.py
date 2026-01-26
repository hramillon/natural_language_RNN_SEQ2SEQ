import numpy as np
import nltk
import tensorflow as tf
from tensorflow import keras
from nltk.corpus import treebank

# -------------------------
# 1. Chargement du corpus
# -------------------------
nltk.download("treebank")
sentences = [[w.lower() for w in sent] for sent in treebank.sents()]

# Limiter le vocabulaire
MAX_VOCAB = 5000
keras.backend.clear_session()

tokenizer = keras.preprocessing.text.Tokenizer(
    num_words=MAX_VOCAB,
    filters="",     # on garde la ponctuation
    lower=True
)
tokenizer.fit_on_texts([" ".join(s) for s in sentences])

word_index = tokenizer.word_index
index_word = {v: k for k, v in word_index.items()}

# -------------------------
# 2. Création des séquences
# -------------------------
SEQ_LEN = 10

X, y = [], []

for sent in sentences:
    seq = tokenizer.texts_to_sequences([sent])[0]
    for i in range(len(seq) - SEQ_LEN):
        X.append(seq[i:i+SEQ_LEN])
        y.append(seq[i+1:i+SEQ_LEN+1])

X = np.array(X)
y = np.array(y)

vocab_size = min(MAX_VOCAB, len(word_index) + 1)

# -------------------------
# 3. Modèle LSTM
# -------------------------
model = keras.Sequential([
    keras.layers.Embedding(vocab_size, 64),
    keras.layers.LSTM(128, return_sequences=True),
    keras.layers.Dense(vocab_size)
])

model.compile(
    optimizer="adam",
    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True)
)

model.fit(X, y, epochs=10, batch_size=64)

# -------------------------
# 4. Génération de texte
# -------------------------
def sample_with_temperature(logits, temperature=1.0):
    logits = logits / temperature
    probs = np.exp(logits) / np.sum(np.exp(logits))
    return np.random.choice(len(probs), p=probs)

def generate_text(seed, length=20, temperature=0.8):
    result = seed[:]
    for _ in range(length):
        seq = tokenizer.texts_to_sequences([result[-SEQ_LEN:]])[0]
        seq = keras.preprocessing.sequence.pad_sequences(
            [seq], maxlen=SEQ_LEN
        )
        logits = model.predict(seq, verbose=0)[0, -1]
        next_idx = sample_with_temperature(logits, temperature)
        next_word = index_word.get(next_idx, "")
        result.append(next_word)
    return " ".join(result)

# Tests
print(generate_text(["i", "am"], 25))
print(generate_text(["the", "company"], 25))
print(generate_text(["he", "said"], 25))
