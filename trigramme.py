import nltk
import numpy as np
from tensorflow import keras

nltk.download('punkt')
nltk.download('treebank')
from nltk.corpus import treebank

sentences = [[w.lower() for w in sent] for sent in treebank.sents()]

X_words = []
y_words = []

for sent in sentences:
    for i in range(len(sent) - 2):  
        X_words.append(sent[i:i+2]) 
        y_words.append(sent[i + 2])   

all_sentences = [" ".join(sent) for sent in sentences]
tokenizer = keras.preprocessing.text.Tokenizer(oov_token="<unk>")
tokenizer.fit_on_texts(all_sentences)

def encode_sequence(seq):
    indices = []
    for w in seq:
        idx = tokenizer.word_index.get(w, tokenizer.word_index[tokenizer.oov_token])
        indices.append(idx)
    return indices

X_encoded = np.array([encode_sequence(seq) for seq in X_words])

y_encoded = np.array([
    tokenizer.word_index.get(w, tokenizer.word_index[tokenizer.oov_token]) 
    for w in y_words
])

vocab_size = len(tokenizer.word_index) + 1
embed_dim = 50

model = keras.Sequential([
    keras.layers.Embedding(input_dim=vocab_size, output_dim=embed_dim, input_length=2),
    keras.layers.SimpleRNN(128),
    keras.layers.Dense(vocab_size)
])

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True)
)

history = model.fit(X_encoded, y_encoded, epochs=10, batch_size=64, verbose=1)

# Calculer la perplexité
split_idx = int(len(X_encoded) * 0.95)

X_train = X_encoded[:split_idx]
y_train = y_encoded[:split_idx]

X_test = X_encoded[split_idx:]
y_test = y_encoded[split_idx:]

train_loss = model.evaluate(X_train, y_train, verbose=0)
test_loss = model.evaluate(X_test, y_test, verbose=0)

train_perplexity = np.exp(train_loss)
test_perplexity = np.exp(test_loss)

print(f"Train Perplexité : {train_perplexity:.2f}")
print(f"Test Perplexité : {test_perplexity:.2f}")

model.save('models/trigram3.keras')