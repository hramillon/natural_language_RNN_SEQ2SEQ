import nltk
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers

try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')
try:
    nltk.data.find('corpora/treebank')
except LookupError:
    nltk.download('treebank')

from nltk.corpus import treebank

# Préparer les données

sentences = [[w.lower() for w in sent] for sent in treebank.sents()]
all_sentences = [" ".join(sent) for sent in sentences]

tokenizer = keras.preprocessing.text.Tokenizer(oov_token="<unk>")
tokenizer.fit_on_texts([s + " <eos>" for s in all_sentences])
vocab_size = len(tokenizer.word_index) + 1

window_size = 5
X_words = []
y_words = []

for sent in sentences:
    sent = sent + ["<eos>"]
    for i in range(len(sent) - window_size):
        X_words.append(sent[i:i+window_size])
        y_words.append(sent[i + window_size])

def encode_sequence(seq):
    return [tokenizer.word_index.get(w, tokenizer.word_index[tokenizer.oov_token]) 
            for w in seq]

X_encoded = np.array([encode_sequence(seq) for seq in X_words])
y_encoded = np.array([tokenizer.word_index.get(w, tokenizer.word_index[tokenizer.oov_token]) 
                       for w in y_words])

embed_dim = 100
temp_model = keras.Sequential([
    layers.Embedding(input_dim=vocab_size, output_dim=embed_dim, input_length=window_size),
    layers.Flatten(),
    layers.Dense(vocab_size)
])

temp_model.compile(
    optimizer=keras.optimizers.Adam(0.001),
    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=['accuracy']
)

temp_model.fit(X_encoded, y_encoded, epochs=15, batch_size=128, verbose=1)
# juste prendre embedding
embedding_layer = temp_model.layers[0]
embedding_model = keras.Sequential([embedding_layer])

embedding_model.save('models/embedding.keras')
print(f"Embedding sauvegardé : vocab_size={vocab_size}, embed_dim={embed_dim}")