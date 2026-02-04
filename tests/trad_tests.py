import numpy as np
import pandas as pd
import re
from tensorflow import keras
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
import pickle

# Load model
model = keras.models.load_model('../models/seq2seq_best.h5')

# Load data
df = pd.read_csv(
    "../ressources/trad_mid.csv",
    sep="\t",
    header=None,
    names=["fr", "ang"]
)

def add_spaces_around_punctuation(text):
    """Ajoute des espaces autour de la ponctuation"""
    text = re.sub(r'\s+([.,?!";()\[\]\-$€£])', r'\1', text)
    text = re.sub(r'([.,?!";()\[\]\-$€£])\s*', r' \1 ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def reverse_sentence(text):
    """Inverse l'ordre des mots"""
    words = text.split()
    return " ".join(reversed(words))

# Preprocessing
df["fr"] = df["fr"].apply(add_spaces_around_punctuation)
df["ang"] = df["ang"].apply(add_spaces_around_punctuation)
df["fr"] = df["fr"].apply(reverse_sentence)

# Split train/val/test
df_train, df_tmp = train_test_split(df, test_size=0.2, random_state=42)
df_val, df_test = train_test_split(df_tmp, test_size=0.5, random_state=42)

print(f"Train set: {len(df_train)} | Val set: {len(df_val)} | Test set: {len(df_test)}")

# Add special tokens
for df_split in [df_train, df_val, df_test]:
    df_split["fr"] = df_split["fr"] + " <end>"
    df_split["ang"] = "<start> " + df_split["ang"]

# Tokenization
tokenizer_fr = Tokenizer(oov_token="<unk>", filters="")
tokenizer_en = Tokenizer(oov_token="<unk>", filters="")
tokenizer_fr.fit_on_texts(df_train["fr"])
tokenizer_en.fit_on_texts(df_train["ang"])

def encode(tokenizer, texts):
    """Convertit les textes en sequences de tokens"""
    return tokenizer.texts_to_sequences(texts)

# Encoding
X_fr_train = encode(tokenizer_fr, df_train["fr"])
X_en_train = encode(tokenizer_en, df_train["ang"])
X_fr_val = encode(tokenizer_fr, df_val["fr"])
X_en_val = encode(tokenizer_en, df_val["ang"])
X_fr_test = encode(tokenizer_fr, df_test["fr"])
X_en_test = encode(tokenizer_en, df_test["ang"])

vocab_fr = len(tokenizer_fr.word_index) + 1
vocab_en = len(tokenizer_en.word_index) + 1

print(f"Vocab FR: {vocab_fr} | Vocab EN: {vocab_en}")

# Padding
max_len_fr = max(len(s) for s in X_fr_train)
max_len_en = max(len(s) for s in X_en_train)

print(f"Max length FR: {max_len_fr} | Max length EN: {max_len_en}")

X_fr_train = pad_sequences(X_fr_train, maxlen=max_len_fr, padding="post")
X_en_train = pad_sequences(X_en_train, maxlen=max_len_en, padding="post")
X_fr_val = pad_sequences(X_fr_val, maxlen=max_len_fr, padding="post")
X_en_val = pad_sequences(X_en_val, maxlen=max_len_en, padding="post")
X_fr_test = pad_sequences(X_fr_test, maxlen=max_len_fr, padding="post")
X_en_test = pad_sequences(X_en_test, maxlen=max_len_en, padding="post")

# Decoder input/output
X_dec_in_train = X_en_train[:, :-1]
y_dec_out_train = X_en_train[:, 1:]
X_dec_in_val = X_en_val[:, :-1]
y_dec_out_val = X_en_val[:, 1:]
X_dec_in_test = X_en_test[:, :-1]
y_dec_out_test = X_en_test[:, 1:]

# Train loss
eval_train = model.evaluate([X_fr_train, X_dec_in_train], y_dec_out_train, verbose=0)
train_loss = eval_train[0]
print(f"Train Loss: {train_loss:.4f}")

# Val loss
eval_val = model.evaluate([X_fr_val, X_dec_in_val], y_dec_out_val, verbose=0)
val_loss = eval_val[0]
print(f"Val Loss: {val_loss:.4f}")

# Test loss
eval_test = model.evaluate([X_fr_test, X_dec_in_test], y_dec_out_test, verbose=0)
test_loss = eval_test[0]
print(f"Test Loss: {test_loss:.4f}")

def translate(sentence):
    """Traduit une phrase du français vers l'anglais"""
    sentence_preprocessed = add_spaces_around_punctuation(sentence.lower())
    sentence_reversed = reverse_sentence(sentence_preprocessed)
    sentence_final = sentence_reversed + " <end>"
    
    # Encoder
    tokens = tokenizer_fr.texts_to_sequences([sentence_final])[0]
    tokens = pad_sequences([tokens], maxlen=max_len_fr, padding="post")
    
    translation = []
    decoder_input = np.zeros((1, max_len_en - 1))
    decoder_input[0, 0] = tokenizer_en.word_index.get("<start>", 1)
    
    # Decoder
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

# Test sentences
test_sentences = [
    "Je suis fatigué",
    "il aime les chats",
    "le football il a changé"
]

for idx, sent in enumerate(test_sentences, 1):
    result = translate(sent)
    print(f"\nTest {idx}:")
    print(f"  FR (original):     {result['original']}")
    print(f"  FR (preprocessing):{result['preprocessed']}")
    print(f"  FR (reversed):     {result['reversed']}")
    print(f"  EN (traduction):   {result['translation']}")