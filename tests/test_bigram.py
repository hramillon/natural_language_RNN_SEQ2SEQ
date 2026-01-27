import nltk
import numpy as np
from tensorflow import keras
import pickle
from collections import Counter

nltk.download('treebank')
from nltk.corpus import treebank

model = keras.models.load_model('../models/bigram.keras')
with open('../models/vocab.pkl', 'rb') as f:
    vocab = pickle.load(f)

reverse_vocab = {v: k for k, v in vocab.items()}
unk_idx = 0

def predict_next(word, top_k=5):
    word = word.lower()
    word_idx = vocab.get(word, unk_idx)
    
    logits = model.predict(np.array([[word_idx]]), verbose=0)
    probs = keras.activations.softmax(logits[0]).numpy()
    
    top_indices = np.argsort(probs)[-top_k:][::-1]
    results = []
    for pred_idx in top_indices:
        pred_word = reverse_vocab.get(pred_idx, "<unk>")
        prob = probs[pred_idx]
        results.append((pred_word, prob))
    
    return results

print("TESTS DE PRÉDICTION")

test_words = ['the', 'i', 'is', 'and', 'to', 'of']
for word in test_words:
    predictions = predict_next(word, top_k=5)
    print(f"\n'{word}' Top 5 prédictions:")
    for i, (pred_word, prob) in enumerate(predictions, 1):
        print(f"  {i}. {pred_word:15s} (prob: {prob:.4f})")

print("\n")
print("ÉVALUATION SUR TEST SET")

sentences = [[w.lower() for w in sent] for sent in treebank.sents()]

word_counts = Counter()
for sent in sentences:
    word_counts.update(sent)

vocab_size = len(vocab) + 1

# Split train/test
train_size = int(0.9 * len(sentences))
test_sents = sentences[train_size:]

X_test = []
y_test = []

for sent in test_sents:
    sent_with_eos = sent + ["<eos>"]
    for i in range(len(sent_with_eos) - 1):
        word_idx = vocab.get(sent_with_eos[i], unk_idx)
        X_test.append(word_idx)
        next_word_idx = vocab.get(sent_with_eos[i + 1], unk_idx)
        y_test.append(next_word_idx)

X_test = np.array(X_test)
y_test = np.array(y_test)

# Évaluer
test_loss = model.evaluate(X_test, y_test, verbose=0)
test_perplexity = np.exp(test_loss)

print(f"\nTest Loss : {test_loss:.4f}")
print(f"Test Perplexité : {test_perplexity:.2f}")