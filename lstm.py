import numpy as np
import nltk
import tensorflow as tf
from tensorflow import keras
from keras.callbacks import EarlyStopping, ModelCheckpoint
from nltk.corpus import treebank

nltk.download("treebank")
MIN_FREQ = 2
keras.backend.clear_session()

sentences = [[w.lower() for w in sent] for sent in treebank.sents()]
sentences_with_eos = [sent + ["<eos>"] for sent in sentences]
tokenizer = keras.preprocessing.text.Tokenizer(
    filters="",
    lower=True,
    oov_token="<unk>"
)

tokenizer.fit_on_texts([" ".join(s) for s in sentences_with_eos])

#filtrer par fréquence
word_counts = tokenizer.word_counts
filtered_word_index = {
    word: idx 
    for word, idx in tokenizer.word_index.items() 
    if tokenizer.word_counts.get(word, 0) >= MIN_FREQ
}
filtered_word_index = {
    word: new_idx 
    for new_idx, (word, _) in enumerate(sorted(filtered_word_index.items(), key=lambda x: x[1]), start=1)
}

word_index = tokenizer.word_index
index_word = {v: k for k, v in word_index.items()}

#context window size
SEQ_LEN = 10

X, y = [], []
for sent in sentences:
    sent_with_eos = sent + ["<eos>"]
    seq = tokenizer.texts_to_sequences([" ".join(sent_with_eos)])[0]
    
    for i in range(len(seq) - SEQ_LEN):
        X.append(seq[i:i+SEQ_LEN])
        y.append(seq[i+1:i+SEQ_LEN+1])

X = np.array(X)
y = np.array(y)
vocab_size = len(word_index) + 1

print(f"word_index length: {len(word_index)}")
print(f"vocab_size: {vocab_size}")
print(f"Index de <eos>: {word_index.get('<eos>')}")
print(f"Index de <unk>: {word_index.get('<unk>')}")

model = keras.Sequential([
    keras.layers.Embedding(vocab_size, 64),
    keras.layers.LSTM(64, return_sequences=True, dropout=0.4, kernel_regularizer=keras.regularizers.l2(0.001)),
    keras.layers.Dropout(0.2),
    keras.layers.Dense(vocab_size)
])

model.compile(
    optimizer="adam",
    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True)
)

checkpoint = ModelCheckpoint('models/lstm.keras', monitor='val_loss', save_best_only=True)
early_stop = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

model.fit(
    X, y,
    epochs=50,
    batch_size=128,
    validation_split=0.2,
    callbacks=[checkpoint, early_stop],
    verbose=1
)


model.save('models/lstm.keras')