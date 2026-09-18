import json
import numpy as np
import re
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Conv1D, GlobalMaxPooling1D, Bidirectional, LSTM, Dense, Dropout,Input,MaxPooling1D,Concatenate
from tensorflow.keras.initializers import Constant
from transformers import pipeline
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

with open("HumanVsGPT-4.json", "r", encoding="utf-8") as f:

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
# 2. Preprocess Text
# ----------------------------
def clean_text(text):
    text = re.sub(r"[^a-zA-Z0-9\s]", "", text)
    return text.lower()

texts = [clean_text(t) for t in texts]

max_words = 20000
tokenizer = Tokenizer(num_words=max_words)
tokenizer.fit_on_texts(texts)
sequences = tokenizer.texts_to_sequences(texts)
maxlen = 300
word_index = tokenizer.word_index

X = pad_sequences(sequences, maxlen=maxlen)
y = np.array(labels)

# ----------------------------
# 3. Load GloVe Embeddings
# ----------------------------
embedding_dim = 100
embeddings_index = {}
with open("glove.6B.100d.txt", encoding="utf-8") as f:
    for line in f:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype="float32")
        embeddings_index[word] = coefs

embedding_matrix = np.zeros((len(tokenizer.word_index) + 1, embedding_dim))
for word, i in tokenizer.word_index.items():
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        embedding_matrix[i] = embedding_vector

# ----------------------------
# 4. Build CNN + BiLSTM Model
# ----------------------------
'''
model = Sequential()
model.add(Embedding(len(tokenizer.word_index) + 1,
                    embedding_dim,
                    embeddings_initializer=Constant(embedding_matrix),
                    input_length=maxlen,
                    trainable=False))
model.add(Conv1D(128, 5, activation="relu"))
model.add(GlobalMaxPooling1D())
model.add(Bidirectional(LSTM(64)))
model.add(Dropout(0.5))
model.add(Dense(1, activation="sigmoid"))
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
#model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

# ----------------------------
# 5. Train/Test Split
# ----------------------------
X_train, X_test, y_train, y_test, texts_train, texts_test = train_test_split(
    X, y, texts, test_size=0.2, random_state=42
)

#model.fit(X_train, y_train, validation_split=0.1, epochs=5, batch_size=32)

model = build_model()
#model.fit([X_train_seq, X_train_syn], y_train, epochs=5, batch_size=32, validation_split=0.1)
model.fit(X_train, y_train, validation_split=0.1, epochs=5, batch_size=32)

# ----------------------------
# 6. Evaluation Function
# ----------------------------
def evaluate_model(X, y_true, dataset_name):
    y_pred_prob = model.predict(X).ravel()
    y_pred = (y_pred_prob > 0.5).astype(int)
    results = {
        "dataset": dataset_name,
        "acc": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred),
        "recall": recall_score(y_true, y_pred),
        "f1": f1_score(y_true, y_pred),
        "auc": roc_auc_score(y_true, y_pred_prob)
    }
    return results

# Evaluate on Original Test Set
results_original = evaluate_model(X_test, y_test, "Original")

# ----------------------------
# 7. Paraphrase Test Set
# ----------------------------
#paraphraser = pipeline("text2text-generation", model="Vamsi/T5_Paraphrase_Paws")

from transformers import T5ForConditionalGeneration, T5Tokenizer
paraphraser_tokenizer = T5Tokenizer.from_pretrained("t5-base")
paraphraser = T5ForConditionalGeneration.from_pretrained("t5-base")
paraphrased_texts = []
for t in texts_test[:300]:  # paraphrase first 100 to save time
    try:
        paraphrase = paraphraser("paraphrase: " + t, max_length=200, num_return_sequences=1)[0]['generated_text']
        paraphrased_texts.append(paraphrase)
    except:
        paraphrased_texts.append(t)  # fallback to original if error

# Convert paraphrased to sequences
paraphrased_seq = tokenizer.texts_to_sequences(paraphrased_texts)
X_paraphrased = pad_sequences(paraphrased_seq, maxlen=maxlen)

# Evaluate on Paraphrased Test Set
results_paraphrased = evaluate_model(X_paraphrased, y_test[:len(paraphrased_texts)], "Paraphrased")

# ----------------------------
# 8. Compare Results
# ----------------------------
df_results = pd.DataFrame([results_original, results_paraphrased])
print(df_results)
