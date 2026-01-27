import nltk
import numpy as np
from tensorflow import keras
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from collections import Counter

try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')
try:
    nltk.data.find('corpora/treebank')
except LookupError:
    nltk.download('treebank')

from nltk.corpus import treebank

# Préparation des données
sentences = [[w.lower() for w in sent] for sent in treebank.sents()]
word_counts = Counter()
for sent in sentences:
    word_counts.update(sent)

MIN_FREQ = 5
filtered_sentences = []
for sent in sentences:
    filtered_sent = [w if word_counts[w] >= MIN_FREQ else "<unk>" for w in sent]
    filtered_sentences.append(filtered_sent)

all_sentences = [" ".join(sent) for sent in filtered_sentences]
tokenizer = keras.preprocessing.text.Tokenizer(oov_token="<unk>")
tokenizer.fit_on_texts([s + " <eos>" for s in all_sentences])
vocab_size = len(tokenizer.word_index) + 1

print(f"Vocabulaire original : {len(word_counts)}")
print(f"Vocabulaire filtré (freq >= {MIN_FREQ}) : {vocab_size - 1}")
print(f"Mots supprimés : {len(word_counts) - (vocab_size - 1)}")

window_size = 1

X_words = []
y_words = []

for sent in filtered_sentences:
    sent_with_eos = sent + ["<eos>"]
    for i in range(len(sent_with_eos) - window_size):
        X_words.append(sent_with_eos[i:i+window_size])
        y_words.append(sent_with_eos[i + window_size])

def encode_sequence(seq):
    return [tokenizer.word_index.get(w, tokenizer.word_index[tokenizer.oov_token]) 
            for w in seq]

X_encoded = np.array([encode_sequence(seq) for seq in X_words])
y_encoded = np.array([tokenizer.word_index.get(w, tokenizer.word_index[tokenizer.oov_token]) 
                      for w in y_words])

embed_dim = 50

from keras import regularizers

model = keras.Sequential([
    keras.layers.Embedding(vocab_size, embed_dim, input_length=window_size),
    keras.layers.Flatten(),
    keras.layers.Dense(128, activation='relu', 
                      kernel_regularizer=regularizers.l2(0.01)),
    keras.layers.Dense(vocab_size)
])

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=['accuracy']
)

callbacks = [
    EarlyStopping(
        monitor='val_loss',
        patience=5,
        restore_best_weights=True
    ),
    ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=3,
        min_lr=1e-6
    )
]

train_size = int(0.9 * len(X_encoded))
X_train, X_val = X_encoded[:train_size], X_encoded[train_size:]
y_train, y_val = y_encoded[:train_size], y_encoded[train_size:]

history = model.fit(
    X_train, y_train,
    epochs=50, 
    batch_size=128,
    validation_data=(X_val, y_val),
    callbacks=callbacks,
    verbose=1
)

# Sauvegarder le modèle et le tokenizer
model.save('models/bigram.keras')

import pickle
with open('models/tokenizer.pkl', 'wb') as f:
    pickle.dump(tokenizer, f)

val_loss, val_acc = model.evaluate(X_val, y_val, verbose=0)
val_perplexity = np.exp(val_loss)
print(f"\nValidation Loss: {val_loss:.4f}")
print(f"Validation Accuracy: {val_acc:.4f}")
print(f"Validation Perplexité: {val_perplexity:.2f}")