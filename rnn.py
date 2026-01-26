import nltk
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers

nltk.download('punkt')
nltk.download('treebank')
from nltk.corpus import treebank

sentences = [[w.lower() for w in sent] for sent in treebank.sents()]
all_sentences = [" ".join(sent) for sent in sentences]

# okenizer avec OOV et ajout d'un token de fin de phrase <eos>
tokenizer = keras.preprocessing.text.Tokenizer(oov_token="<unk>")
tokenizer.fit_on_texts([s + " <eos>" for s in all_sentences])

vocab_size = len(tokenizer.word_index) + 1

window_size = 5  # nombre de mots considérés pour le RNN
X_words = []
y_words = []

for sent in sentences:
    sent = sent + ["<eos>"]  # ajouter fin de phrase
    for i in range(len(sent) - window_size):
        X_words.append(sent[i:i+window_size])
        y_words.append(sent[i + window_size])

def encode_sequence(seq):
    return [tokenizer.word_index.get(w, tokenizer.word_index[tokenizer.oov_token]) for w in seq]

X_encoded = np.array([encode_sequence(seq) for seq in X_words])
y_encoded = np.array([tokenizer.word_index.get(w, tokenizer.word_index[tokenizer.oov_token]) for w in y_words])

embed_dim = 100
rnn_units = 128

model = keras.Sequential([
    layers.Embedding(input_dim=vocab_size, output_dim=embed_dim, input_length=window_size),
    layers.SimpleRNN(rnn_units, activation='tanh'),
    layers.Dense(vocab_size)
])

model.compile(
    optimizer=keras.optimizers.Adam(0.001),
    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True)
)

model.fit(X_encoded, y_encoded, epochs=10, batch_size=128, verbose=1)

reverse_index = {v: k for k, v in tokenizer.word_index.items()}

def generate_sentence_rnn(start_words, max_length=20):
    sentence = start_words.copy()
    for _ in range(max_length):
        # Encoder les derniers mots
        seq = [tokenizer.word_index.get(w, tokenizer.word_index["<unk>"]) for w in sentence[-window_size:]]
        seq = np.array([seq])
        # Prédire le mot suivant
        logits = model.predict(seq, verbose=0)
        pred_idx = np.argmax(logits)
        next_word = reverse_index.get(pred_idx, "<unk>")
        if next_word == "<eos>":
            break
        sentence.append(next_word)
    return " ".join(sentence)


print(generate_sentence_rnn(["i", "am"]))
print(generate_sentence_rnn(["the", "company"]))
print(generate_sentence_rnn(["he", "said"]))
