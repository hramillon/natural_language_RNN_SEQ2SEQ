import nltk
import numpy as np
from tensorflow import keras
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer

nltk.download('treebank')
from nltk.corpus import treebank

model = keras.models.load_model('../models/trigram3.keras')

sentences = [[w.lower() for w in sent] for sent in treebank.sents()]
all_sentences = [" ".join(sent) for sent in sentences]
tokenizer = Tokenizer(oov_token="<unk>")
tokenizer.fit_on_texts(all_sentences)

reverse_index = {v: k for k, v in tokenizer.word_index.items()}
unk_token_idx = tokenizer.word_index["<unk>"]

window_size = 2

def generate_sentence_rnn(start_words, max_length=20, top_k=5):
    sentence = start_words.copy()
    for _ in range(max_length):
        seq = [tokenizer.word_index.get(w, unk_token_idx) for w in sentence[-window_size:]]
        seq = np.array([seq])
        
        logits = model.predict(seq, verbose=0)
        probs = tf.nn.softmax(logits[0]).numpy()
        
        # Prendre les top-k indices (ignorer les faibles probabilités)
        top_k_indices = np.argsort(probs)[-top_k:][::-1]
        top_k_probs = probs[top_k_indices]
        
        # Normaliser et choisir aléatoirement parmi top-k
        top_k_probs = top_k_probs / top_k_probs.sum()
        pred_idx = np.random.choice(top_k_indices, p=top_k_probs)
        
        next_word = reverse_index.get(pred_idx, "<unk>")
        if next_word == "<unk>" or next_word == "<eos>":
            break
        sentence.append(next_word)
    return " ".join(sentence)

print(generate_sentence_rnn(["i", "am"]))
print(generate_sentence_rnn(["the", "cat"]))
print(generate_sentence_rnn(["he", "said"]))