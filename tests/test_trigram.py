import nltk
import numpy as np
from tensorflow import keras
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer

nltk.download('treebank')
from nltk.corpus import treebank

model = keras.models.load_model('../models/trigram3.keras')

# Recréer le tokenizer
sentences = [[w.lower() for w in sent] for sent in treebank.sents()]
all_sentences = [" ".join(sent) for sent in sentences]
tokenizer = Tokenizer(oov_token="<unk>")
tokenizer.fit_on_texts(all_sentences)

reverse_index = {v: k for k, v in tokenizer.word_index.items()}
unk_token_idx = tokenizer.word_index["<unk>"]

def encode_sequence(seq):
    indices = []
    for w in seq:
        idx = tokenizer.word_index.get(w, unk_token_idx)
        indices.append(idx)
    return indices

def predict_next(word1, word2, top_k=5):
    word1 = word1.lower()
    word2 = word2.lower()
    idx1 = tokenizer.word_index.get(word1, unk_token_idx)
    idx2 = tokenizer.word_index.get(word2, unk_token_idx)
    logits = model.predict(np.array([[idx1, idx2]]), verbose=0)
    probs = keras.activations.softmax(logits[0]).numpy()
    top_indices = np.argsort(probs)[-top_k:][::-1]
    results = []
    for pred_idx in top_indices:
        pred_word = reverse_index.get(pred_idx, "<unk>")
        prob = probs[pred_idx]
        results.append((pred_word, prob))
    return results

print("TESTS DE PRÉDICTION TRIGRAMME")

test_pairs = [('the', 'cat'), ('i', 'am'), ('you', 'are'), ('he', 'is'), ('and', 'the')]

for word1, word2 in test_pairs:
    predictions = predict_next(word1, word2, top_k=5)
    print(f"\n'{word1} {word2}' Top 5 prédictions:")
    for i, (pred_word, prob) in enumerate(predictions, 1):
        print(f"  {i}. {pred_word:15s} (prob: {prob:.4f})")

print("ÉVALUATION SUR TEST SET")

X_words = []
y_words = []
for sent in sentences:
    for i in range(len(sent) - 2):
        X_words.append(sent[i:i+2])
        y_words.append(sent[i + 2])

X_encoded = np.array([encode_sequence(seq) for seq in X_words])
y_encoded = np.array([
    tokenizer.word_index.get(w, unk_token_idx) 
    for w in y_words
])

split_idx = int(len(X_encoded) * 0.9)
X_test = X_encoded[split_idx:]
y_test = y_encoded[split_idx:]


# Évaluer
test_loss = model.evaluate(X_test, y_test, verbose=0)
test_perplexity = np.exp(test_loss)

print(f"\nTest Loss : {test_loss:.4f}")
print(f"Test Perplexité : {test_perplexity:.2f}")