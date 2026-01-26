import nltk
import numpy as np
from tensorflow import keras

nltk.download('punkt')
nltk.download('treebank')
from nltk.corpus import treebank

# Tous les fichiers
fileids = treebank.fileids()

split = int(len(fileids) * 0.95)
train_fileids = fileids[:split]
test_fileids = fileids[split:]

train_sents = treebank.sents(fileids=train_fileids)
test_sents = treebank.sents(fileids=test_fileids)

# Charger les phrases tokenisées et mettre en minuscules
sentences = [[w.lower() for w in sent] for sent in train_sents]

X_words = []
y_words = []
for sent in sentences:
    for i in range(len(sent) - 1):
        X_words.append(sent[i])
        y_words.append(sent[i + 1])

# Créer le tokenizer avec OOV token
all_sentences = [" ".join(sent) for sent in sentences]
tokenizer = keras.preprocessing.text.Tokenizer(oov_token="<unk>")
tokenizer.fit_on_texts(all_sentences)

# Transformer X_words et y_words en indices
X_encoded = np.array([tokenizer.texts_to_sequences([w])[0][0] if tokenizer.texts_to_sequences([w]) and len(tokenizer.texts_to_sequences([w])[0]) > 0 else tokenizer.word_index[tokenizer.oov_token] for w in X_words])
y_encoded = np.array([tokenizer.texts_to_sequences([w])[0][0] if tokenizer.texts_to_sequences([w]) and len(tokenizer.texts_to_sequences([w])[0]) > 0 else tokenizer.word_index[tokenizer.oov_token] for w in y_words])

vocab_size = len(tokenizer.word_index) + 1

embed_dim = 50
model = keras.Sequential([
    keras.layers.Embedding(input_dim=vocab_size, output_dim=embed_dim, input_length=1),
    keras.layers.Flatten(),
    keras.layers.Dense(vocab_size)
])

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True)
)

model.fit(X_encoded, y_encoded, epochs=10, batch_size=128)

reverse_index = {v: k for k, v in tokenizer.word_index.items()}

def predict_next(word):
    word = word.lower()
    idx = tokenizer.word_index.get(word, tokenizer.word_index["<unk>"])
    logits = model.predict(np.array([idx]), verbose=0)
    pred_idx = np.argmax(logits)
    return reverse_index.get(pred_idx, "<unk>")

print("I →", predict_next("I"))
print("you →", predict_next("you"))
print("is →", predict_next("is"))


test_sentences = [[w.lower() for w in sent] for sent in test_sents]

X_test_words = []
y_test_words = []
for sent in test_sentences:
    for i in range(len(sent) - 1):
        X_test_words.append(sent[i])
        y_test_words.append(sent[i + 1])

X_encoded = np.array([tokenizer.texts_to_sequences([w])[0][0] if tokenizer.texts_to_sequences([w]) and len(tokenizer.texts_to_sequences([w])[0]) > 0 else tokenizer.word_index[tokenizer.oov_token] for w in X_test_words])
y_encoded = np.array([tokenizer.texts_to_sequences([w])[0][0] if tokenizer.texts_to_sequences([w]) and len(tokenizer.texts_to_sequences([w])[0]) > 0 else tokenizer.word_index[tokenizer.oov_token] for w in y_test_words])

# Évaluer
test_loss = model.evaluate(X_encoded, y_encoded, verbose=0)
test_perplexity = np.exp(test_loss)
print(f"Test Perplexité : {test_perplexity:.2f}")