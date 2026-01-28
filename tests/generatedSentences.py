import nltk
import numpy as np
from tensorflow import keras
import tensorflow as tf
from collections import Counter

nltk.download('treebank')
from nltk.corpus import treebank

model = keras.models.load_model('../models/gru.keras')

# IMPORTANT : reproduire EXACTEMENT le preprocessing d'entraînement
MIN_FREQ = 5

def clean_sentence(sent):
    forbidden = {'-NONE-', "''", '``', '--', '$', ';', ','}
    return [w for w in sent 
            if '*' not in w 
            and not any(char.isdigit() for char in w)
            and w not in forbidden]

sentences = [[w.lower() for w in clean_sentence(sent)] for sent in treebank.sents()]
sentences = [s for s in sentences if len(s) > 2]

# Compter les fréquences EXACTEMENT comme à l'entraînement
word_counts = Counter()
for sent in sentences:
    word_counts.update(sent)

frequent_words = {word for word, count in word_counts.items() if count >= MIN_FREQ}
frequent_words.add("<eos>")
frequent_words.add("<unk>")

# Créer word_to_idx comme à l'entraînement
word_to_idx = {word: idx for idx, word in enumerate(sorted(frequent_words))}
idx_to_word = {idx: word for word, idx in word_to_idx.items()}

vocab_size = len(word_to_idx)
print(f"Vocab size: {vocab_size}")
print(f"Model expects vocab size: {model.layers[0].input_dim}")

SEQ_LEN = 5  # À adapter à ta valeur d'entraînement

def generate_sentence(start_words, max_length=20, top_k=5):
    sentence = start_words.copy()
    
    for _ in range(max_length):
        context = sentence[-SEQ_LEN:]
        while len(context) < SEQ_LEN:
            context = ["<unk>"] + context
        
        seq = [word_to_idx.get(w, word_to_idx["<unk>"]) for w in context]
        seq = np.array([seq])
        
        logits = model.predict(seq, verbose=0)
        last_logits = logits[0, -1, :]
        probs = tf.nn.softmax(last_logits).numpy()
        
        top_k_indices = np.argsort(probs)[-top_k:][::-1]
        top_k_probs = probs[top_k_indices]
        top_k_probs = top_k_probs / top_k_probs.sum()
        
        pred_idx = np.random.choice(top_k_indices, p=top_k_probs)
        next_word = idx_to_word.get(pred_idx, "<unk>")
        
        rank = 0
        while next_word == "<unk>" and rank < len(top_k_indices) - 1:
            rank += 1
            pred_idx = top_k_indices[rank]
            next_word = idx_to_word.get(pred_idx, "<unk>")
        
        if next_word == "<eos>":
            break
        
        if next_word == "<unk>":
            break
        
        sentence.append(next_word)
    
    return " ".join(sentence)

print("\nGraine: ['i', 'am']")
print(generate_sentence(["i", "am"]))

print("\nGraine: ['the', 'cat']")
print(generate_sentence(["the", "cat"]))

print("\nGraine: ['he', 'said']")
print(generate_sentence(["he", "said"]))

print("\nGraine: ['the']")
print(generate_sentence(["the"]))