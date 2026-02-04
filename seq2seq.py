import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, Input, Model
from tensorflow.keras.layers import Input, Embedding, GRU, Dense, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
import re

df = pd.read_csv(
    "ressources/trad_mid.csv",
    sep="\t",
    header=None,
    names=["fr", "ang"]
)

def add_spaces_around_punctuation(text):
    text = re.sub(r'\s+([.,?!";()\[\]\-$€£])', r'\1', text)
    text = re.sub(r'([.,?!";()\[\]\-$€£])\s*', r' \1 ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def reverse_sentence(text):
    words = text.split()
    return " ".join(reversed(words))

df["fr"] = df["fr"].apply(add_spaces_around_punctuation)
df["ang"] = df["ang"].apply(add_spaces_around_punctuation)

df["fr"] = df["fr"].apply(reverse_sentence)

# Split train/val/test
df_train, df_tmp = train_test_split(df, test_size=0.2, random_state=42)
df_val, df_test = train_test_split(df_tmp, test_size=0.5, random_state=42)

# Ajouter les tokens spéciaux
for df_split in [df_train, df_val, df_test]:
    df_split["fr"] = df_split["fr"] + " <end>"
    df_split["ang"] = "<start> " + df_split["ang"]

# Tokenization "unk" is useless,we didn't put a limit
tokenizer_fr = Tokenizer(oov_token="<unk>", filters="")
tokenizer_en = Tokenizer(oov_token="<unk>", filters="")

tokenizer_fr.fit_on_texts(df_train["fr"])
tokenizer_en.fit_on_texts(df_train["ang"])

#transforme nos colonnes de phrases de mots en colonnes de phrases de token
def encode(tokenizer, texts):
    return tokenizer.texts_to_sequences(texts)

X_fr_train = encode(tokenizer_fr, df_train["fr"])
X_fr_val = encode(tokenizer_fr, df_val["fr"])
X_fr_test = encode(tokenizer_fr, df_test["fr"])

X_en_train = encode(tokenizer_en, df_train["ang"])
X_en_val = encode(tokenizer_en, df_val["ang"])
X_en_test = encode(tokenizer_en, df_test["ang"])

vocab_fr = len(tokenizer_fr.word_index) + 1
vocab_en = len(tokenizer_en.word_index) + 1

# Padding
max_len_fr = max(len(s) for s in X_fr_train)
max_len_en = max(len(s) for s in X_en_train)

#toutes le sphrases ont la même taille , max_len, ou les mots manguants sont remplacés par 0?
X_fr_train = pad_sequences(X_fr_train, maxlen=max_len_fr, padding="post")
X_fr_val = pad_sequences(X_fr_val, maxlen=max_len_fr, padding="post")
X_fr_test = pad_sequences(X_fr_test, maxlen=max_len_fr, padding="post")

X_en_train = pad_sequences(X_en_train, maxlen=max_len_en, padding="post")
X_en_val = pad_sequences(X_en_val, maxlen=max_len_en, padding="post")
X_en_test = pad_sequences(X_en_test, maxlen=max_len_en, padding="post")

X_dec_in_train = X_en_train[:, :-1]
y_dec_out_train = X_en_train[:, 1:]

X_dec_in_val = X_en_val[:, :-1]
y_dec_out_val = X_en_val[:, 1:]

X_dec_in_test = X_en_test[:, :-1]
y_dec_out_test = X_en_test[:, 1:]

#on fait en Functional API car donne plus de liberté c'est plus claire pour faire des seq2seq
# commençons l'encoder
enc_inputs = Input(shape=(max_len_fr,), name="encoder_inputs")
enc_embed = layers.Embedding(vocab_fr, 128, name="encoder_embedding")(enc_inputs)
enc_x = layers.Dropout(0.2)(enc_embed)

# Couche GRU
enc_outputs, enc_state_fwd, enc_state_bwd = layers.Bidirectional(
    layers.GRU(128, return_sequences=True, return_state=True),
    name="encoder_bi_gru"
)(enc_x)
# enc_outputs: (batch_size, max_len_fr, 256)
# enc_state_fwd, enc_state_bwd: (batch_size, 128)

# Decoder
dec_inputs = Input(shape=(max_len_en - 1,), name="decoder_inputs")
dec_embed = layers.Embedding(vocab_en, 128, mask_zero=True, name="decoder_embedding")(dec_inputs)
dec_x = layers.Dropout(0.2)(dec_embed)

dec_initial_state = layers.Concatenate(name="combine_encoder_states")([enc_state_fwd, enc_state_bwd])

# 1er GRU
dec_x, _ = layers.GRU(
    256,
    return_sequences=True, 
    return_state=True,
    name="decoder_gru_1"
)(dec_x, initial_state=dec_initial_state)
dec_x = layers.Dropout(0.2)(dec_x)

# Query: vient du decoder
# Key/Value: vient de l'encoder
attention = layers.MultiHeadAttention(
    num_heads=4,
    key_dim=32,
    dropout=0.2,
    name="attention"
)(
    query=dec_x,
    value=enc_outputs,
    key=enc_outputs
)

# Ajouter une connexion résiduelle + layer norm
dec_x = layers.Add(name="attention_add")([dec_x, attention])
dec_x = layers.LayerNormalization(epsilon=1e-6, name="attention_norm")(dec_x)

dec_x = layers.Dropout(0.2)(dec_x)

# 2nd GRU
dec_x, _ = layers.GRU(
    256, 
    return_sequences=True, 
    return_state=True,
    name="decoder_gru_2"
)(dec_x)
dec_x = layers.Dropout(0.2)(dec_x)

# output
logits = layers.Dense(vocab_en, name="output_dense")(dec_x)

model = Model([enc_inputs, dec_inputs], logits, name="seq2seq_attention")
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001, clipnorm=1.0),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=['accuracy']
)

# Callbacks
checkpoint = ModelCheckpoint(
    'models/seq2seq_best.h5',
    monitor='val_loss',
    save_best_only=True,
    verbose=1
)
reduce_lr = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=1,
    min_lr=0.00001,
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
    batch_size=32,
    epochs=100,
    callbacks=[checkpoint, reduce_lr, early_stopping]
)

def translate(sentence):
    sentence_preprocessed = add_spaces_around_punctuation(sentence.lower())
    
    sentence_reversed = reverse_sentence(sentence_preprocessed)
    
    sentence_final = sentence_reversed + " <end>"
    
    tokens = tokenizer_fr.texts_to_sequences([sentence_final])[0]
    tokens = pad_sequences([tokens], maxlen=max_len_fr, padding="post")
    
    translation = []
    decoder_input = np.zeros((1, max_len_en - 1))
    decoder_input[0, 0] = tokenizer_en.word_index.get("<start>", 1)
    
    for i in range(1, max_len_en - 1):
        predictions = model.predict([tokens, decoder_input], verbose=0)
        predicted_id = np.argmax(predictions[0, i, :])
        
        if predicted_id == tokenizer_en.word_index.get("<end>", 2):
            break
        
        if predicted_id > 0:
            word = tokenizer_en.index_word.get(predicted_id, "<unk>")
            translation.append(word)
            decoder_input[0, i] = predicted_id
    
    translation_text = " ".join(translation)
    
    return {
        "original": sentence,
        "preprocessed": sentence_preprocessed,
        "reversed": sentence_reversed,
        "translation": translation_text
    }

test_sentences = [
    df_test["fr"].iloc[0].replace("<start> ", "").replace(" <end>", ""),
    df_test["fr"].iloc[1].replace("<start> ", "").replace(" <end>", ""),
    df_test["fr"].iloc[2].replace("<start> ", "").replace(" <end>", ""),
]

for idx, sent in enumerate(test_sentences, 1):
    
    result = translate(sent)
    
    print(f"\nfr:")
    print(f"   {result['original']}")
    
    print(f"\nfr2")
    print(f"   {result['preprocessed']}")
    
    print(f"\nfr3")
    print(f"   {result['reversed']}")
    
    print(f"\nen:")
    print(f"   {result['translation']}")