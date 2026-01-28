import numpy as np
import nltk
import tensorflow as tf
from tensorflow import keras
from nltk.corpus import treebank

nltk.download("treebank")
sentences = [[w.lower() for w in sent] for sent in treebank.sents()]

MIN_FREQ = 2

keras.backend.clear_session()
tokenizer = keras.preprocessing.text.Tokenizer(
    filters="",
    lower=True,
    oov_token="<unk>"
)

tokenizer.word_index = {"<unk>": 1, "<eos>": 2}
tokenizer.fit_on_texts([" ".join(s) for s in sentences])

word_index = tokenizer.word_index
index_word = {v: k for k, v in word_index.items()}

#context window size
SEQ_LEN = 10

X, y = [], []
for sent in sentences:
    sent_with_eos = sent + ["<eos>"]
    seq = tokenizer.texts_to_sequences([" ".join(sent_with_eos)])[0]
    
    for i in range(len(seq) - SEQ_LEN):
        X.append(seq[i:i+SEQ_LEN])
        y.append(seq[i+1:i+SEQ_LEN+1])

X = np.array(X)
y = np.array(y)
vocab_size = len(word_index) + 1

print(f"Vocab size: {vocab_size}")
print(f"Index de <eos>: {word_index.get('<eos>', 'non trouvé')}")

model = keras.Sequential([
    keras.layers.Embedding(vocab_size, 64),
    keras.layers.LSTM(128, return_sequences=True, dropout=0.2),
    keras.layers.LSTM(64, return_sequences=False, dropout=0.2),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dropout(0.2),
    keras.layers.Dense(vocab_size)
])

model.compile(
    optimizer="adam",
    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True)
)

model.fit(X, y, epochs=10, batch_size=64)

