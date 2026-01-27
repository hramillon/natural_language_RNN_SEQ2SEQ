import nltk
import numpy as np
from tensorflow import keras
from keras.models import load_model
import pickle

try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')
try:
    nltk.data.find('corpora/treebank')
except LookupError:
    nltk.download('treebank')

from nltk.corpus import treebank

model = load_model('../models/bigram.keras')

with open('../models/tokenizer.pkl', 'rb') as f:
    tokenizer = pickle.load(f)

vocab_size = len(tokenizer.word_index) + 1
reverse_index = {v: k for k, v in tokenizer.word_index.items()}

def predict_next(word, top_k=5):
    word = word.lower()
    idx = tokenizer.word_index.get(word, tokenizer.word_index["<unk>"])
    
    logits = model.predict(np.array([[idx]]), verbose=0)
    
    probs = keras.activations.softmax(logits[0]).numpy()
    
    top_indices = np.argsort(probs)[-top_k:][::-1]
    
    results = []
    for pred_idx in top_indices:
        word_pred = reverse_index.get(pred_idx, "<unk>")
        prob = probs[pred_idx]
        results.append((word_pred, prob))
    
    return results

print("\n" + "="*50)
print("TESTS DE PRÉDICTION")
print("="*50)

test_words = ['the', 'i', 'is', 'and', 'to', 'of']

for word in test_words:
    predictions = predict_next(word, top_k=5)
    print(f"\n'{word}'Top 5 prédictions:")
    for i, (pred_word, prob) in enumerate(predictions, 1):
        print(f"  {i}. {pred_word:15s} (prob: {prob:.4f})")

print("\n" + "="*50)
print("ÉVALUATION SUR TEST SET")
print("="*50)

sentences = [[w.lower() for w in sent] for sent in treebank.sents()]
from collections import Counter
word_counts = Counter()
for sent in sentences:
    word_counts.update(sent)

MIN_FREQ = 4
filtered_sentences = []
for sent in sentences:
    filtered_sent = [w if word_counts[w] >= MIN_FREQ else "<unk>" for w in sent]
    filtered_sentences.append(filtered_sent)

train_size = int(0.9 * len(filtered_sentences))
test_sents = filtered_sentences[train_size:]

X_test_words = []
y_test_words = []

for sent in test_sents:
    sent_with_eos = sent + ["<eos>"]
    for i in range(len(sent_with_eos) - 1):
        X_test_words.append([sent_with_eos[i]])  
        y_test_words.append(sent_with_eos[i + 1])

def encode_sequence(seq):
    return [tokenizer.word_index.get(w, tokenizer.word_index[tokenizer.oov_token]) 
            for w in seq]

X_test_encoded = np.array([encode_sequence(seq) for seq in X_test_words])
y_test_encoded = np.array([tokenizer.word_index.get(w, tokenizer.word_index[tokenizer.oov_token]) 
                           for w in y_test_words])

print(f"Nombre d'exemples test : {len(X_test_encoded)}")

test_loss, test_acc = model.evaluate(X_test_encoded, y_test_encoded, verbose=0)
test_perplexity = np.exp(test_loss)

print(f"\nTest Loss : {test_loss:.4f}")
print(f"Test Accuracy : {test_acc:.4f}")
print(f"Test Perplexité : {test_perplexity:.2f}")