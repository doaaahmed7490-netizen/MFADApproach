import json
import numpy as np
import pandas as pd
import re
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Conv1D, GlobalMaxPooling1D, Bidirectional, LSTM, Dense, Dropout,Input,MaxPooling1D,Concatenate
from tensorflow.keras.initializers import Constant
from tensorflow.keras.models import Model, load_model


# ----------------------------
# 1. Load HC3 Dataset
# ----------------------------

#with open("IEEE-ChatGPT-GenerationWithHuman", "r", encoding="utf-8") as f:

#with open("HumanVsYi-Large.json", "r", encoding="utf-8") as f:
#with open("HumanVsMistral.json", "r", encoding="utf-8") as f:
#with open("HumanVsgGemma.json", "r", encoding="utf-8") as f:
#with open("HumanVsllama.json", "r", encoding="utf-8") as f:
#with open("HumanVsQwene.json", "r", encoding="utf-8") as f:

with open("TweetHumanvsLlama.json", "r", encoding="utf-8") as f:

    data = json.load(f)

texts, labels = [], []

for item in data:
    for ans in item["Human"]:
        texts.append(ans)
        labels.append(0)  # human
    for ans in item["GPT-4"]:
        texts.append(ans)
        labels.append(1)  # AI
# ----------------------------
# 2. Text Preprocessing
# ----------------------------
def clean_text(text):
    text = re.sub(r"[^a-zA-Z0-9\s]", "", text)
    return text.lower()

texts = [clean_text(t) for t in texts]

# Tokenization
max_words = 20000
tokenizer = Tokenizer(num_words=max_words)
tokenizer.fit_on_texts(texts)
sequences = tokenizer.texts_to_sequences(texts)
word_index = tokenizer.word_index

maxlen = 300  # pad/truncate length
X = pad_sequences(sequences, maxlen=maxlen)
y = np.array(labels)

# ----------------------------
# 3. Load GloVe embeddings
# ----------------------------
embedding_dim = 100
embeddings_index = {}
#with open("glove.6B.100d.txt", encoding="utf-8") as f:
with open("G:\Code\glove.6B.100d.txt", encoding='utf-8') as f:

    for line in f:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype="float32")
        embeddings_index[word] = coefs

embedding_matrix = np.zeros((len(word_index) + 1, embedding_dim))
for word, i in word_index.items():
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        embedding_matrix[i] = embedding_vector

# ----------------------------
# 4. Build CNN + BiLSTM Model
# ----------------------------
'''
model = Sequential()
model.add(Embedding(len(word_index) + 1,
                    embedding_dim,
                    embeddings_initializer=Constant(embedding_matrix),
                    input_length=maxlen,
                    trainable=False))
model.add(Conv1D(128, 5, activation="relu"))
model.add(MaxPooling1D())
model.add(Bidirectional(LSTM(64)))

model.add(Dropout(0.5))
model.add(Dense(1, activation="sigmoid"))

model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
'''
def build_model():
    text_input = Input(shape=(300,))

    embedding = Embedding(input_dim=len(word_index)+1,
                          output_dim=embedding_dim,
                          weights=[embedding_matrix],
                          input_length=300,
                          trainable=False)(text_input)

    conv = Conv1D(128, 5, activation='relu')(embedding)
    pool = MaxPooling1D(pool_size=2)(conv)
    #pool = GlobalMaxPooling1D()(conv)

    bilstm = Bidirectional(LSTM(64))(pool)

    #merged = Concatenate()([bilstm, meta_input])
    merged = Concatenate(name="concat_features")([bilstm])

   # merged = Concatenate()([pool, meta_input])
    dense = Dense(64, activation='relu')(merged)
    dropout = Dropout(0.5)(dense)
    output = Dense(1, activation='sigmoid')(dropout)

    model = Model(inputs=[text_input], outputs=output)
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model
# ----------------------------
# 5. Train/Test Split
# ----------------------------
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = build_model()

model.fit(X_train, y_train, validation_split=0.1, epochs=5, batch_size=32)

# ----------------------------
# 6. Robustness Evaluation by Length
# ----------------------------
y_pred_prob = model.predict(X_test).ravel()
y_pred = (y_pred_prob > 0.5).astype(int)

# bucket evaluation
test_texts = [texts[i] for i in range(len(X)) if i in list(range(len(X)))[len(X_train):]]
lengths = [len(t.split()) for t in test_texts]

results = []
for bucket_name, (low, high) in {
    "short (<=50)": (0, 50),
    "medium (51-100)": (51, 100),
    "long (>100)": (101, float("inf"))
}.items():
    idx = [i for i, l in enumerate(lengths) if low <= l <= high]
    if not idx:
        continue
    y_true_bucket = y_test[idx]
    y_pred_bucket = y_pred[idx]
    y_prob_bucket = y_pred_prob[idx]
    results.append({
        "bucket": bucket_name,
        "acc": accuracy_score(y_true_bucket, y_pred_bucket),
        "precision": precision_score(y_true_bucket, y_pred_bucket),
        "recall": recall_score(y_true_bucket, y_pred_bucket),
        "f1": f1_score(y_true_bucket, y_pred_bucket),
        "auc": roc_auc_score(y_true_bucket, y_prob_bucket)
    })

df_results = pd.DataFrame(results)
print(df_results)
