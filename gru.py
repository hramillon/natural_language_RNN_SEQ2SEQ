import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, GRU, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
import pickle

with open('ressources/penn_treebank_subset.txt', 'r', encoding='utf-8') as file:
    text_data = file.read()

tokenizer = Tokenizer()
tokenizer.fit_on_texts([text_data])
total_words = len(tokenizer.word_index) + 1

input_sequences = []
for line in text_data.split('\n'):
    if line.strip():
        token_list = tokenizer.texts_to_sequences([line])[0]
        for i in range(1, len(token_list)):
            input_sequences.append(token_list[:i+1])

max_sequence_len = 20
input_sequences = np.array(pad_sequences(input_sequences, maxlen=max_sequence_len, padding='pre'))
X, labels = input_sequences[:, :-1], input_sequences[:, -1]
y = tf.keras.utils.to_categorical(labels, num_classes=total_words)

model = Sequential([
    Embedding(total_words, 100, input_length=max_sequence_len-1),
    Dropout(0.2),
    GRU(150, return_sequences=True),
    Dropout(0.2),
    GRU(100),
    Dropout(0.2),
    Dense(total_words, activation='softmax')
])

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

early_stop = EarlyStopping(monitor='loss', patience=5, restore_best_weights=True)
model.fit(X, y, batch_size=128, epochs=100, verbose=1, callbacks=[early_stop], validation_split=0.1)

model.save('models/gru.keras')
with open('models/tokenizer.pickle', 'wb') as handle:
    pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)
