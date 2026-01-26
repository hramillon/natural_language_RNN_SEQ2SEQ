import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Embedding, GRU, Dense, Input
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
import pickle

# Charger le dataset
df = pd.read_csv(
    "trad.csv",
    sep="\t",
    header=None,
    names=["fr", "ang"]
)

# Ne garder que les colonnes utiles
df = df[["fr", "ang"]]

# Nettoyage de base
df["fr"] = df["fr"].astype(str).str.lower()
df["ang"] = df["ang"].astype(str).str.lower()

# Supprimer doublons exacts
df = df.drop_duplicates(subset=["fr", "ang"])
df = df.reset_index(drop=True)

print("Paires uniques :", len(df))

N_TOTAL = 20000  # recommandé CPU

if len(df) > N_TOTAL:
    df = df.sample(n=N_TOTAL, random_state=42)

print("Après échantillonnage :", len(df))

df_train, df_tmp = train_test_split(df, test_size=0.2, random_state=42)
df_val, df_test = train_test_split(df_tmp, test_size=0.5, random_state=42)

print("Train :", len(df_train))
print("Val   :", len(df_val))
print("Test  :", len(df_test))

df_train["fr"] = "<start> " + df_train["fr"] + " <end>"
df_train["ang"] = "<start> " + df_train["ang"] + " <end>"

df_val["fr"] = "<start> " + df_val["fr"] + " <end>"
df_val["ang"] = "<start> " + df_val["ang"] + " <end>"

df_test["fr"] = "<start> " + df_test["fr"] + " <end>"
df_test["ang"] = "<start> " + df_test["ang"] + " <end>"

tokenizer_fr = Tokenizer(oov_token="<unk>", filters="")
tokenizer_en = Tokenizer(oov_token="<unk>", filters="")

tokenizer_fr.fit_on_texts(df_train["fr"])
tokenizer_en.fit_on_texts(df_train["ang"])

def encode(tokenizer, texts):
    return tokenizer.texts_to_sequences(texts)

X_fr_train = encode(tokenizer_fr, df_train["fr"])
X_fr_val   = encode(tokenizer_fr, df_val["fr"])

X_en_train = encode(tokenizer_en, df_train["ang"])
X_en_val   = encode(tokenizer_en, df_val["ang"])

vocab_fr = len(tokenizer_fr.word_index) + 1
vocab_en = len(tokenizer_en.word_index) + 1

max_len_fr = max(len(s) for s in X_fr_train)
max_len_en = max(len(s) for s in X_en_train)

X_fr_train = pad_sequences(X_fr_train, maxlen=max_len_fr, padding="post")
X_fr_val   = pad_sequences(X_fr_val,   maxlen=max_len_fr, padding="post")

X_en_train = pad_sequences(X_en_train, maxlen=max_len_en, padding="post")
X_en_val   = pad_sequences(X_en_val,   maxlen=max_len_en, padding="post")

# Decoder input / output
X_dec_in_train = X_en_train[:, :-1]
y_dec_out_train = X_en_train[:, 1:]

X_dec_in_val = X_en_val[:, :-1]
y_dec_out_val = X_en_val[:, 1:]

embed_dim = 64
rnn_units = 128

# Encoder
enc_inputs = Input(shape=(max_len_fr,))
enc_embed = Embedding(vocab_fr, embed_dim)(enc_inputs)
_, enc_state = GRU(rnn_units, return_state=True)(enc_embed)

# Decoder
dec_inputs = Input(shape=(max_len_en - 1,))
dec_embed = Embedding(vocab_en, embed_dim)(dec_inputs)
dec_outputs = GRU(rnn_units, return_sequences=True)(
    dec_embed, initial_state=enc_state
)

logits = Dense(vocab_en)(dec_outputs)

model = Model([enc_inputs, dec_inputs], logits)
model.compile(
    optimizer="adam",
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
)

model.summary()

history = model.fit(
    [X_fr_train, X_dec_in_train],
    y_dec_out_train,
    validation_data=([X_fr_val, X_dec_in_val], y_dec_out_val),
    batch_size=32,
    epochs=20
)

model.save("models/seq2seq_fr_en.keras")

with open("models/tokenizer_fr.pkl", "wb") as f:
    pickle.dump(tokenizer_fr, f)

with open("models/tokenizer_en.pkl", "wb") as f:
    pickle.dump(tokenizer_en, f)


index_en = {v: k for k, v in tokenizer_en.word_index.items()}

def translate(sentence, max_len=20):
    sentence = "<start> " + sentence.lower() + " <end>"
    seq = tokenizer_fr.texts_to_sequences([sentence])
    seq = pad_sequences(seq, maxlen=max_len_fr, padding="post")

    dec_input = np.array([[tokenizer_en.word_index["<start>"]]])
    result = []

    state = model.layers[2](model.layers[1](seq))[1]

    for _ in range(max_len):
        emb = model.layers[4](dec_input)
        out, state = model.layers[5](emb, initial_state=state)
        logits = model.layers[6](out)
        token = np.argmax(logits[0, -1])

        if token == tokenizer_en.word_index.get("<end>"):
            break

        result.append(index_en.get(token, "<unk>"))
        dec_input = np.array([[token]])

    return " ".join(result)
