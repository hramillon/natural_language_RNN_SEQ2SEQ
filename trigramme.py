import nltk
import numpy as np
from tensorflow import keras

nltk.download('punkt')
nltk.download('treebank')
from nltk.corpus import treebank

sentences = [[w.lower() for w in sent] for sent in treebank.sents()]

X_words = []  # Séquences de 2 mots
y_words = []  # Le mot suivant

for sent in sentences:
    for i in range(len(sent) - 2):  
        X_words.append(sent[i:i+2]) 
        y_words.append(sent[i + 2])   

# Créer le tokenizer avec OOV token
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
    keras.layers.Flatten(),
    keras.layers.Dense(vocab_size)
])

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True)
)

# Entraînement
history = model.fit(X_encoded, y_encoded, epochs=10, batch_size=128, verbose=1)

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

reverse_index = {v: k for k, v in tokenizer.word_index.items()}

def predict_next(word1, word2):
    word1 = word1.lower()
    word2 = word2.lower()
    
    idx1 = tokenizer.word_index.get(word1, tokenizer.word_index[tokenizer.oov_token])
    idx2 = tokenizer.word_index.get(word2, tokenizer.word_index[tokenizer.oov_token])
    
    logits = model.predict(np.array([[idx1, idx2]]), verbose=0)
    pred_idx = np.argmax(logits)
    
    return reverse_index.get(pred_idx, "<unk>")

# Exemples de prédiction
print("\nPrédictions trigramme :")
print("I am →", predict_next("i", "am"))
print("you are →", predict_next("you", "are"))
print("he is →", predict_next("he", "is"))