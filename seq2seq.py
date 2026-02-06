import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, Input, Model
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
import re
import sentencepiece as spm

# gestion des tokens et transformation phrase pour être utilisé par seq2seq
df = pd.read_csv(
    "ressources/trad_max_cleaned.csv",
    sep="\t",
    header=None,
    names=["fr", "ang"]
)

def add_spaces_around_punctuation(text):
    text = re.sub(r'\s+([.,?!";()\[\]\-$€£])', r'\1', text)
    text = re.sub(r'([.,?!";()\[\]\-$€£])\s*', r' \1 ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

df["fr"] = df["fr"].apply(add_spaces_around_punctuation)
df["ang"] = df["ang"].apply(add_spaces_around_punctuation)

df_train, df_tmp = train_test_split(df, test_size=0.2, random_state=42)
df_val, df_test = train_test_split(df_tmp, test_size=0.5, random_state=42)

sp_fr = spm.SentencePieceProcessor(model_file="models/sp_fr.model")
sp_en = spm.SentencePieceProcessor(model_file="models/sp_en.model")

vocab_fr = sp_fr.vocab_size()
vocab_en = sp_en.vocab_size()

def encode_with_sentencepiece_fr(sp, texts):
    encoded = [sp.EncodeAsIds(text) for text in texts]
    encoded = [seq + [2] for seq in encoded]
    return encoded

def encode_with_sentencepiece_en(sp, texts):
    encoded = [sp.EncodeAsIds(text) for text in texts]
    encoded = [[1] + seq + [2] for seq in encoded]
    return encoded

X_fr_train = encode_with_sentencepiece_fr(sp_fr, df_train["fr"].tolist())
X_en_train = encode_with_sentencepiece_en(sp_en, df_train["ang"].tolist())

X_fr_val = encode_with_sentencepiece_fr(sp_fr, df_val["fr"].tolist())
X_en_val = encode_with_sentencepiece_en(sp_en, df_val["ang"].tolist())

X_fr_test = encode_with_sentencepiece_fr(sp_fr, df_test["fr"].tolist())
X_en_test = encode_with_sentencepiece_en(sp_en, df_test["ang"].tolist())

lengths_fr = [len(s) for s in X_fr_train]
lengths_en = [len(s) for s in X_en_train]

max_len_fr = int(np.percentile(lengths_fr, 99))
max_len_en = int(np.percentile(lengths_en, 99))

X_fr_train = pad_sequences(X_fr_train, maxlen=max_len_fr, padding="post")
X_en_train = pad_sequences(X_en_train, maxlen=max_len_en, padding="post")

X_fr_val = pad_sequences(X_fr_val, maxlen=max_len_fr, padding="post")
X_en_val = pad_sequences(X_en_val, maxlen=max_len_en, padding="post")

X_fr_test = pad_sequences(X_fr_test, maxlen=max_len_fr, padding="post")
X_en_test = pad_sequences(X_en_test, maxlen=max_len_en, padding="post")

X_dec_in_train = X_en_train[:, :-1]
y_dec_out_train = X_en_train[:, 1:]

X_dec_in_val = X_en_val[:, :-1]
y_dec_out_val = X_en_val[:, 1:]

X_dec_in_test = X_en_test[:, :-1]
y_dec_out_test = X_en_test[:, 1:]

#on fait en Functional API car c'est plus claire pour faire des seq2seq
#Encoder
enc_inputs = Input(shape=(max_len_fr,), name="encoder_inputs")
enc_embed = layers.Embedding(vocab_fr, 256, name="encoder_embedding")(enc_inputs)
enc_x = layers.Dropout(0.1)(enc_embed)

enc_outputs, enc_state = layers.GRU(
    256, 
    return_sequences=True, 
    return_state=True,
    name="encoder_gru"
)(enc_x)

# Decoder
dec_inputs = Input(shape=(max_len_en - 1,), name="decoder_inputs")
dec_embed = layers.Embedding(vocab_en, 256, mask_zero=True, name="decoder_embedding")(dec_inputs)
dec_x = layers.Dropout(0.1)(dec_embed)

dec_x, _ = layers.GRU(
    256,
    return_sequences=True,
    return_state=True,
    name="decoder_gru"
)(dec_x, initial_state=enc_state)

# Attention simple
attention = layers.AdditiveAttention(name="attention")(
    [dec_x, enc_outputs]
)
dec_x = layers.Concatenate(name="concat_attention")([dec_x, attention])

logits = layers.Dense(vocab_en, name="output_dense")(dec_x)

model = Model([enc_inputs, dec_inputs], logits, name="seq2seq_simple")

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.0005, clipnorm=1.0),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=['accuracy']
)

checkpoint = ModelCheckpoint(
    'models/final_translation2.h5',
    monitor='val_loss',
    save_best_only=True,
    verbose=1
)

reduce_lr = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=2,
    min_lr=0.0001,
    verbose=1
)

early_stopping = EarlyStopping(
    monitor='val_loss',
    patience=3,
    restore_best_weights=True,
    verbose=1
)

history = model.fit(
    [X_fr_train, X_dec_in_train],
    y_dec_out_train,
    validation_data=([X_fr_val, X_dec_in_val], y_dec_out_val),
    batch_size=256,
    epochs=100,
    callbacks=[checkpoint, reduce_lr, early_stopping],
    verbose=1
)