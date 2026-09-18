# ============================================================
# MFAD: Multi-Feature Accurate Detection
# CNN + BiLSTM + Handcrafted Syntactic/Statistical Features
# FEATURE-LEVEL SHAP ONLY
#
# Main runtime improvement:
# 1. Text-level SHAP has been completely removed.
# 2. Handcrafted features are extracted once and cached.
# 3. SHAP changes ONLY the lightweight named handcrafted features.
# 4. SHAP model predictions are batched.
# 5. Feature extraction progress is displayed.
# ============================================================

import os
import json
import re
import random
import hashlib
from pathlib import Path
from collections import Counter

import numpy as np
import pandas as pd
import nltk
import textstat
import matplotlib.pyplot as plt
import shap
import tensorflow as tf

from nltk import word_tokenize, pos_tag
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.util import ngrams

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
    roc_curve,
    auc,
)


from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input,
    Embedding,
    Conv1D,
    MaxPooling1D,
    Bidirectional,
    LSTM,
    Dense,
    Concatenate,
    Dropout,
)
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences


# ============================================================
# 1. REPRODUCIBILITY
# ============================================================

SEED = 42

np.random.seed(SEED)
random.seed(SEED)
tf.random.set_seed(SEED)


# ============================================================
# 2. PARAMETERS
# ============================================================

MAX_LEN = 300
EMBEDDING_DIM = 100
BATCH_SIZE = 32
EPOCHS = 5

GLOVE_PATH = r"G:\Code\glove.6B.100d.txt"
MODEL_PATH = r"G:\Code\Casestudy1\human_ai_classifier_interpretable.keras"
#DATASET_PATH = "HumanVsGPT-4.json"
DATASET_PATH = "IEEE-ChatGPT-GenerationWithHuman.json"
# Feature cache.
# After the first feature-extraction run, later runs will reuse it.
FEATURE_CACHE_PATH = "MFAD_handcrafted_features_cache_fast.csv"

# SHAP parameters.
# 15 lightweight features only, so this is much faster than text SHAP.
SHAP_BACKGROUND_SIZE = 20
SHAP_NSAMPLES = 50



# ============================================================
# 3. FEATURE NAMES
# ============================================================

FEATURE_NAMES = [
    "Total Words",
    "Total Sentences",
    "Total Unique Words",
    "Type-Token Ratio",
    "Total Stop Words",
    "Total Punctuation",
    "Total Discourse Markers",
    "Readability Score",
    "Syllable Count",
    "Average Sentence Length",
    "Average Word Length",
    "POS Tag Distribution",
    "Sentence Complexity",
    "Top Bigram Frequency",
    "Top Trigram Frequency",
]


# ============================================================
# 4. NLTK RESOURCES
# ============================================================

for resource in [
    "punkt",
    "punkt_tab",
    "averaged_perceptron_tagger",
    "averaged_perceptron_tagger_eng",
    "stopwords",
    "wordnet",
]:
    try:
        nltk.download(resource, quiet=True)
    except Exception as exc:
        print(f"Warning: could not download NLTK resource '{resource}': {exc}")


# ============================================================
# 5. DATASET
# ============================================================

def load_hc3_json(path):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    texts = []
    labels = []

    for entry in data:
        for h in entry.get("Human", []):
            if h and isinstance(h, str):
                texts.append(h)
                labels.append(0)

        for a in entry.get("GPT-4", []):
            if a and isinstance(a, str):
                texts.append(a)
                labels.append(1)

    return pd.DataFrame(
        {
            "text": texts,
            "label": labels,
        }
    )


# ============================================================
# 7. TEXT PREPROCESSING
# ============================================================

def clean_text(text):
    if not isinstance(text, str):
        return ""

    text = text.lower()

    text = re.sub(
        r"http\S+|www\S+|https\S+",
        "",
        text,
    )

    text = re.sub(
        r"#\w+",
        "",
        text,
    )

    text = re.sub(
        r"[\$\€\£\₹\¥\₽\₩\¢]+[\d,.]*",
        "",
        text,
    )

    text = re.sub(
        r"[^\w\s]",
        "",
        text,
    )

    text = re.sub(
        r"\s+",
        " ",
        text,
    ).strip()

    tokens = word_tokenize(text)

    tokens = [
        lemmatizer.lemmatize(word)
        for word in tokens
    ]

    return " ".join(tokens)


# ============================================================
# 8. HANDCRAFTED FEATURES
# ============================================================

DISCOURSE_MARKERS = {
    "however",
    "therefore",
    "moreover",
    "furthermore",
    "nevertheless",
    "consequently",
    "thus",
    "hence",
    "additionally",
    "in addition",
    "for example",
    "for instance",
    "in contrast",
    "on the other hand",
    "in conclusion",
    "as a result",
    "firstly",
    "secondly",
    "finally",
}

COMPLEXITY_MARKERS = {
    "because",
    "although",
    "though",
    "while",
    "whereas",
    "which",
    "that",
    "if",
    "when",
    "since",
    "unless",
    "and",
    "or",
    "but",
}


def extract_features(text):
    """
    Extract the lightweight MFAD handcrafted features.

    Important:
    This function is called during dataset preparation and for
    the two SHAP samples only. Text-level SHAP is NOT used.
    """

    if not isinstance(text, str) or not text.strip():
        return [0.0] * len(FEATURE_NAMES)

    tokens = word_tokenize(text)

    word_tokens = [
        token
        for token in tokens
        if token.isalpha()
    ]

    if len(word_tokens) == 0:
        return [0.0] * len(FEATURE_NAMES)

    total_words = len(word_tokens)

    sentences = nltk.sent_tokenize(text)

    total_sentences = len(sentences)

    if total_sentences == 0:
        total_sentences = 1

    lowercase_words = [
        word.lower()
        for word in word_tokens
    ]

    total_unique_words = len(
        set(lowercase_words)
    )

    type_token_ratio = (
        total_unique_words / total_words
        if total_words > 0
        else 0
    )

    total_stop_words = sum(
        1
        for word in lowercase_words
        if word in stop_words
    )

    total_punctuation = sum(
        1
        for char in text
        if char in ".,!?;:'\"-()[]{}"
    )

    text_lower = text.lower()

    total_discourse_markers = 0

    for marker in DISCOURSE_MARKERS:
        pattern = r"\b" + re.escape(marker) + r"\b"

        total_discourse_markers += len(
            re.findall(
                pattern,
                text_lower,
            )
        )

    # --------------------------------------------------------
    # Readability
    # --------------------------------------------------------

    try:
        readability_score = (
            textstat.flesch_reading_ease(text)
        )
    except Exception:
        readability_score = 0

    # --------------------------------------------------------
    # Syllable count
    # --------------------------------------------------------

    try:
        syllable_count = (
            textstat.syllable_count(text)
        )
    except Exception:
        syllable_count = 0

    # --------------------------------------------------------
    # Average sentence length
    # --------------------------------------------------------

    average_sentence_length = (
        total_words / total_sentences
        if total_sentences > 0
        else 0
    )

    # --------------------------------------------------------
    # Average word length
    # --------------------------------------------------------

    average_word_length = float(
        np.mean(
            [
                len(word)
                for word in word_tokens
            ]
        )
    )

    # --------------------------------------------------------
    # POS tag distribution entropy
    # --------------------------------------------------------

    pos_tags = pos_tag(word_tokens)

    pos_counter = Counter(
        tag
        for _, tag in pos_tags
    )

    total_pos_tags = len(pos_tags)

    pos_probabilities = [
        count / total_pos_tags
        for count in pos_counter.values()
    ]

    pos_distribution_entropy = -sum(
        p * np.log2(p)
        for p in pos_probabilities
        if p > 0
    )

    # --------------------------------------------------------
    # Sentence complexity
    # --------------------------------------------------------

    complexity_count = sum(
        1
        for word in lowercase_words
        if word in COMPLEXITY_MARKERS
    )

    sentence_complexity = (
        complexity_count / total_sentences
        if total_sentences > 0
        else 0
    )

    # --------------------------------------------------------
    # Top bigram frequency
    # --------------------------------------------------------

    bigrams = list(
        ngrams(
            lowercase_words,
            2,
        )
    )

    if bigrams:
        bigram_counts = Counter(bigrams)

        top_bigram_frequency = (
            bigram_counts.most_common(1)[0][1]
        )
    else:
        top_bigram_frequency = 0

    # --------------------------------------------------------
    # Top trigram frequency
    # --------------------------------------------------------

    trigrams = list(
        ngrams(
            lowercase_words,
            3,
        )
    )

    if trigrams:
        trigram_counts = Counter(trigrams)

        top_trigram_frequency = (
            trigram_counts.most_common(1)[0][1]
        )
    else:
        top_trigram_frequency = 0

    return [
        total_words,
        total_sentences,
        total_unique_words,
        type_token_ratio,
        total_stop_words,
        total_punctuation,
        total_discourse_markers,
        readability_score,
        syllable_count,
        average_sentence_length,
        average_word_length,
        pos_distribution_entropy,
        sentence_complexity,
        top_bigram_frequency,
        top_trigram_frequency,
    ]


# ============================================================
# 9. FEATURE CACHE
# ============================================================

def _text_hash(text):
    return hashlib.sha256(
        str(text).encode("utf-8")
    ).hexdigest()


def load_feature_cache():
    cache_path = Path(FEATURE_CACHE_PATH)

    if not cache_path.exists():
        return {}

    try:
        cache_df = pd.read_csv(
            cache_path,
            encoding="utf-8",
        )

        required_columns = (
            ["text_hash"] + FEATURE_NAMES
        )

        if not all(
            column in cache_df.columns
            for column in required_columns
        ):
            print(
                "Existing feature cache has an old format. "
                "A new cache will be created."
            )
            return {}

        cache = {}

        for _, row in cache_df.iterrows():
            cache[
                row["text_hash"]
            ] = [
                float(row[feature])
                for feature in FEATURE_NAMES
            ]

        print(
            f"Loaded {len(cache):,} cached feature vectors."
        )

        return cache

    except Exception as exc:
        print(
            f"Could not load feature cache: {exc}"
        )
        return {}


def save_feature_cache(cache):
    rows = []

    for text_hash, values in cache.items():
        row = {
            "text_hash": text_hash
        }

        row.update(
            dict(
                zip(
                    FEATURE_NAMES,
                    values,
                )
            )
        )

        rows.append(row)

    cache_df = pd.DataFrame(rows)

    cache_df.to_csv(
        FEATURE_CACHE_PATH,
        index=False,
        encoding="utf-8",
    )

    print(
        f"Feature cache saved: {FEATURE_CACHE_PATH}"
    )


def extract_features_with_cache(texts):
    """
    Extract features only for texts that are not already cached.

    This is important because Only lightweight handcrafted features are calculated.
    """

    cache = load_feature_cache()

    results = []

    total = len(texts)
    missing = 0

    # Count missing first.
    for text in texts:
        key = _text_hash(text)

        if key not in cache:
            missing += 1

    print(
        f"Total texts: {total:,}"
    )
    print(
        f"Already cached: {total - missing:,}"
    )
    print(
        f"Need extraction: {missing:,}"
    )

    if missing > 0:
        print(
            "\nStarting handcrafted feature extraction..."
        )
        print(
            "This is the only stage that may be slow on the "
            "first run because grammar/spelling are expensive."
        )

        for i, text in enumerate(texts, start=1):
            key = _text_hash(text)

            if key not in cache:
                cache[key] = extract_features(text)

            if i == 1 or i % 100 == 0 or i == total:
                print(
                    f"Feature extraction progress: "
                    f"{i:,}/{total:,} "
                    f"({100.0 * i / total:.1f}%)"
                )

        save_feature_cache(cache)

    for text in texts:
        key = _text_hash(text)
        results.append(cache[key])

    return np.asarray(
        results,
        dtype=np.float32,
    )


# ============================================================
# 10. LOAD DATA
# ============================================================

print("\n============================================================")
print("MFAD START")
print("============================================================")

df = load_hc3_json(DATASET_PATH)

print(
    "Dataset shape:",
    df.shape,
)

print(
    "\nClass distribution:"
)

print(
    df["label"].value_counts()
)


# ============================================================
# 11. CLEAN TEXT
# ============================================================

print(
    "\nCleaning text..."
)

df["text_clean"] = df["text"].apply(
    clean_text
)


# ============================================================
# 12. TRAIN / TEST SPLIT
# ============================================================

train_df, test_df = train_test_split(
    df,
    test_size=0.20,
    random_state=SEED,
    stratify=df["label"],
)

train_df = train_df.reset_index(drop=True)
test_df = test_df.reset_index(drop=True)

print(
    "\nTraining samples:",
    len(train_df),
)

print(
    "Testing samples:",
    len(test_df),
)


# ============================================================
# 13. TOKENIZER
# ============================================================

print(
    "\nFitting tokenizer on training data..."
)

tokenizer = Tokenizer(
    oov_token="<OOV>"
)

tokenizer.fit_on_texts(
    train_df["text_clean"]
)

word_index = tokenizer.word_index

VOCAB_SIZE = len(word_index) + 1

print(
    "Vocabulary size:",
    VOCAB_SIZE,
)


# ============================================================
# 14. TEXT SEQUENCES
# ============================================================

X_train_seq = pad_sequences(
    tokenizer.texts_to_sequences(
        train_df["text_clean"]
    ),
    maxlen=MAX_LEN,
    padding="pre",
    truncating="pre",
)

X_test_seq = pad_sequences(
    tokenizer.texts_to_sequences(
        test_df["text_clean"]
    ),
    maxlen=MAX_LEN,
    padding="pre",
    truncating="pre",
)


# ============================================================
# 15. HANDCRAFTED FEATURES
# ============================================================

print(
    "\n============================================================"
)
print(
    "HANDCRAFTED FEATURE EXTRACTION"
)
print(
    "============================================================"
)

all_texts = pd.concat(
    [
        train_df["text"],
        test_df["text"],
    ],
    ignore_index=True,
)

all_features = extract_features_with_cache(
    all_texts.tolist()
)

X_train_syn = all_features[
    : len(train_df)
]

X_test_syn = all_features[
    len(train_df) :
]

print(
    "\nX_train_syn shape:",
    X_train_syn.shape,
)

print(
    "X_test_syn shape:",
    X_test_syn.shape,
)


# ============================================================
# 16. STANDARDIZE FEATURES
# ============================================================

scaler = StandardScaler()

X_train_syn_scaled = scaler.fit_transform(
    X_train_syn
).astype(np.float32)

X_test_syn_scaled = scaler.transform(
    X_test_syn
).astype(np.float32)


# ============================================================
# 17. LABELS
# ============================================================

y_train = train_df["label"].values.astype(
    np.float32
)

y_test = test_df["label"].values.astype(
    np.float32
)


# ============================================================
# 18. LOAD GLOVE
# ============================================================

print(
    "\n============================================================"
)
print(
    "LOADING GLOVE"
)
print(
    "============================================================"
)

embedding_index = {}

with open(
    GLOVE_PATH,
    encoding="utf-8",
) as f:
    for line in f:
        values = line.rstrip().split()

        if len(values) != EMBEDDING_DIM + 1:
            continue

        word = values[0]

        vector = np.asarray(
            values[1:],
            dtype="float32",
        )

        embedding_index[word] = vector

print(
    "Number of GloVe words:",
    len(embedding_index),
)


# ============================================================
# 19. EMBEDDING MATRIX
# ============================================================

embedding_matrix = np.zeros(
    (
        VOCAB_SIZE,
        EMBEDDING_DIM,
    ),
    dtype=np.float32,
)

for word, index in word_index.items():

    if index >= VOCAB_SIZE:
        continue

    embedding_vector = embedding_index.get(
        word
    )

    if embedding_vector is not None:
        embedding_matrix[index] = (
            embedding_vector
        )


# Free GloVe dictionary after creating the matrix.
del embedding_index


# ============================================================
# 20. BUILD MFAD MODEL
# ============================================================

def build_model():

    text_input = Input(
        shape=(MAX_LEN,),
        name="text_input",
    )

    meta_input = Input(
        shape=(len(FEATURE_NAMES),),
        name="syntactic_input",
    )

    embedding = Embedding(
        input_dim=VOCAB_SIZE,
        output_dim=EMBEDDING_DIM,
        weights=[embedding_matrix],
        trainable=False,
        name="glove_embedding",
    )(text_input)

    conv = Conv1D(
        filters=128,
        kernel_size=5,
        activation="relu",
        name="cnn_layer",
    )(embedding)

    pool = MaxPooling1D(
        pool_size=2,
        name="max_pooling",
    )(conv)

    bilstm = Bidirectional(
        LSTM(64),
        name="bilstm",
    )(pool)

    merged = Concatenate(
        name="concat_features",
    )(
        [
            bilstm,
            meta_input,
        ]
    )

    dense = Dense(
        64,
        activation="relu",
        name="dense_layer",
    )(merged)

    dropout = Dropout(
        0.5,
        name="dropout",
    )(dense)

    output = Dense(
        1,
        activation="sigmoid",
        name="output",
    )(dropout)

    model = Model(
        inputs=[
            text_input,
            meta_input,
        ],
        outputs=output,
    )

    model.compile(
        optimizer=tf.keras.optimizers.Adam(
            learning_rate=0.001
        ),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )

    return model


# ============================================================
# 21. TRAIN
# ============================================================

print(
    "\n============================================================"
)
print(
    "BUILDING MFAD MODEL"
)
print(
    "============================================================"
)

model = build_model()

model.summary()

print(
    "\n============================================================"
)
print(
    "TRAINING MFAD"
)
print(
    "============================================================"
)

history = model.fit(
    [
        X_train_seq,
        X_train_syn_scaled,
    ],
    y_train,
    validation_split=0.10,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    verbose=1,
)


# ============================================================
# 22. SAVE MODEL
# ============================================================

model.save(
    MODEL_PATH
)

print(
    "\nModel saved successfully:"
)
print(
    MODEL_PATH
)


# ============================================================
# 23. MODEL EVALUATION
# ============================================================

print(
    "\n============================================================"
)
print(
    "MODEL EVALUATION"
)
print(
    "============================================================"
)

y_prob = model.predict(
    [
        X_test_seq,
        X_test_syn_scaled,
    ],
    batch_size=256,
    verbose=0,
).ravel()

y_pred = (
    y_prob >= 0.5
).astype(int)

accuracy = accuracy_score(
    y_test,
    y_pred,
)

precision = precision_score(
    y_test,
    y_pred,
    zero_division=0,
)

recall = recall_score(
    y_test,
    y_pred,
    zero_division=0,
)

f1 = f1_score(
    y_test,
    y_pred,
    zero_division=0,
)

roc_auc = roc_auc_score(
    y_test,
    y_prob,
)

print(
    "Accuracy:",
    accuracy,
)

print(
    "Precision:",
    precision,
)

print(
    "Recall:",
    recall,
)

print(
    "F1 Score:",
    f1,
)

print(
    "AUROC:",
    roc_auc,
)

print(
    "\nClassification Report:"
)

print(
    classification_report(
        y_test,
        y_pred,
        target_names=[
            "Human",
            "AI-generated",
        ],
        zero_division=0,
    )
)

cm = confusion_matrix(
    y_test,
    y_pred,
)

print(
    "\nConfusion Matrix:"
)

print(
    cm
)

tn, fp, fn, tp = cm.ravel()

fpr_value = (
    fp / (fp + tn)
    if (fp + tn) > 0
    else 0
)

print(
    "\nFalse Positive Rate:",
    fpr_value,
)


# ============================================================
# 24. ROC CURVE
# ============================================================

fpr, tpr, thresholds = roc_curve(
    y_test,
    y_prob,
)

roc_auc_curve = auc(
    fpr,
    tpr,
)

plt.figure(
    figsize=(8, 6)
)

plt.plot(
    fpr,
    tpr,
    label=f"MFAD (AUC = {roc_auc_curve:.4f})",
)

plt.plot(
    [0, 1],
    [0, 1],
    linestyle="--",
)

plt.xlabel(
    "False Positive Rate"
)

plt.ylabel(
    "True Positive Rate"
)

plt.title(
    "ROC Curve for MFAD"
)

plt.legend()

plt.grid()

plt.tight_layout()

plt.savefig(
    "MFAD_ROC_Curve.png",
    dpi=300,
    bbox_inches="tight",
)

plt.show()

plt.close()


# ============================================================
# 25. FAST MODEL INPUT FOR NORMAL PREDICTION
# ============================================================

def prepare_model_input(texts):
    if isinstance(texts, str):
        texts = [texts]

    texts = list(texts)

    cleaned_texts = [
        clean_text(text)
        for text in texts
    ]

    sequences = pad_sequences(
        tokenizer.texts_to_sequences(
            cleaned_texts
        ),
        maxlen=MAX_LEN,
        padding="pre",
        truncating="pre",
    )

    syntactic_features = np.asarray(
        extract_features_with_cache(
            texts
        ),
        dtype=np.float32,
    )

    syntactic_scaled = scaler.transform(
        syntactic_features
    ).astype(np.float32)

    return (
        sequences,
        syntactic_scaled,
        cleaned_texts,
        syntactic_features,
    )


# ============================================================
# 26. FEATURE-LEVEL SHAP ONLY
# ============================================================

def explain_syntactic_features(
    text,
    sample_name="Sample",
    background_size=20,
    nsamples=50,
):
    """
    SHAP explanation for the lightweight MFAD handcrafted features ONLY.

    The CNN + BiLSTM text sequence is fixed.

    SHAP changes only:
        Total Words
        Total Sentences
        Total Unique Words
        Type-Token Ratio
        Total Stop Words
        Total Punctuation
        Total Discourse Markers
        Readability Score
        Syllable Count
        Average Sentence Length
        Average Word Length
        POS Tag Distribution
        Sentence Complexity
        Top Bigram Frequency
        Top Trigram Frequency

    Therefore there is NO text/token masking and NO repeated
    LanguageTool/SpellChecker calls inside SHAP.
    """

    print(
        "\n============================================================"
    )
    print(
        f"FEATURE-LEVEL SHAP: {sample_name}"
    )
    print(
        "============================================================"
    )

    # --------------------------------------------------------
    # Prepare this sample once.
    # --------------------------------------------------------

    (
        sequences,
        syntactic_scaled,
        _,
        original_features,
    ) = prepare_model_input(
        [text]
    )

    fixed_sequence = sequences[0]

    sample_features = (
        syntactic_scaled[0]
        .reshape(1, -1)
    )

    # --------------------------------------------------------
    # Background data
    # --------------------------------------------------------

    rng = np.random.default_rng(
        SEED
    )

    background_size = min(
        int(background_size),
        len(X_train_syn_scaled),
    )

    background_indices = rng.choice(
        len(X_train_syn_scaled),
        size=background_size,
        replace=False,
    )

    background_data = (
        X_train_syn_scaled[
            background_indices
        ]
    )

    # --------------------------------------------------------
    # Prediction function
    #
    # IMPORTANT:
    # The text sequence NEVER changes.
    # Only the 17 handcrafted feature values change.
    # --------------------------------------------------------

    def predict_features(feature_matrix):

        feature_matrix = np.asarray(
            feature_matrix,
            dtype=np.float32,
        )

        if feature_matrix.ndim == 1:
            feature_matrix = (
                feature_matrix.reshape(
                    1,
                    -1,
                )
            )

        repeated_sequences = np.repeat(
            fixed_sequence.reshape(
                1,
                -1,
            ),
            feature_matrix.shape[0],
            axis=0,
        )

        predictions = model.predict(
            [
                repeated_sequences,
                feature_matrix,
            ],
            batch_size=256,
            verbose=0,
        ).ravel()

        return predictions

    # --------------------------------------------------------
    # Kernel SHAP
    # --------------------------------------------------------

    print(
        f"SHAP background: {background_size}"
    )

    print(
        f"SHAP nsamples: {nsamples}"
    )

    print(
        "Creating feature-only SHAP explainer..."
    )

    explainer = shap.KernelExplainer(
        predict_features,
        background_data,
    )

    print(
        "Calculating SHAP values for 15 features..."
    )

    shap_result = explainer.shap_values(
        sample_features,
        nsamples=nsamples,
        l1_reg="num_features(15)",
    )

    # --------------------------------------------------------
    # SHAP API compatibility
    # --------------------------------------------------------

    if isinstance(shap_result, list):
        shap_result = shap_result[0]

    shap_result = np.asarray(
        shap_result
    )

    if shap_result.ndim == 3:
        shap_result = shap_result[
            :, :,
            0
        ]

    if shap_result.ndim == 1:
        shap_result = shap_result.reshape(
            1,
            -1,
        )

    if (
        shap_result.shape[1]
        != len(FEATURE_NAMES)
    ):
        raise ValueError(
            "Unexpected SHAP feature count: "
            f"{shap_result.shape[1]}. "
            f"Expected {len(FEATURE_NAMES)}."
        )

    # --------------------------------------------------------
    # Feature result table
    # --------------------------------------------------------

    feature_results = pd.DataFrame(
        {
            "Feature": FEATURE_NAMES,
            "Original_Value": original_features[0],
            "Scaled_Value": syntactic_scaled[0],
            "SHAP_Value": shap_result[0],
            "Absolute_SHAP": np.abs(
                shap_result[0]
            ),
        }
    )

    feature_results = (
        feature_results.sort_values(
            "Absolute_SHAP",
            ascending=False,
        )
        .reset_index(drop=True)
    )

    print(
        "\n============================================================"
    )
    print(
        "LOCAL FEATURE SHAP IMPORTANCE"
    )
    print(
        "============================================================"
    )

    print(
        feature_results.to_string(
            index=False
        )
    )

    # --------------------------------------------------------
    # Save CSV
    # --------------------------------------------------------

    filename_csv = (
        f"{sample_name}_Feature_SHAP_Values.csv"
    )

    feature_results.to_csv(
        filename_csv,
        index=False,
        encoding="utf-8",
    )

    print(
        f"\nFeature SHAP CSV saved: {filename_csv}"
    )

    # --------------------------------------------------------
    # SHAP bar plot
    # --------------------------------------------------------

    base_value = float(
        np.asarray(
            explainer.expected_value
        ).reshape(-1)[0]
    )

    explanation = shap.Explanation(
        values=shap_result[0],
        base_values=base_value,
        data=syntactic_scaled[0],
        feature_names=FEATURE_NAMES,
    )

    plt.figure(
        figsize=(11, 8)
    )

    shap.plots.bar(
        explanation,
        max_display=len(FEATURE_NAMES),
        show=False,
    )

    plt.title(
        f"SHAP Importance of MFAD Features - {sample_name}",
        fontsize=14,
    )

    plt.tight_layout()

    filename_plot = (
        f"{sample_name}_Feature_SHAP.png"
    )

    plt.savefig(
        filename_plot,
        dpi=300,
        bbox_inches="tight",
    )

    plt.show()

    plt.close()

    print(
        f"Feature SHAP plot saved: {filename_plot}"
    )

    return (
        shap_result,
        feature_results,
    )


# ============================================================
# 27. SELECT ONE HUMAN AND ONE AI SAMPLE
# ============================================================

human_indices = np.where(
    y_test == 0
)[0]

ai_indices = np.where(
    y_test == 1
)[0]

if len(human_indices) == 0:
    raise ValueError(
        "No human-written sample exists in the test set."
    )

if len(ai_indices) == 0:
    raise ValueError(
        "No AI-generated sample exists in the test set."
    )

human_index = int(
    human_indices[0]
)

ai_index = int(
    ai_indices[0]
)

human_text = test_df.iloc[
    human_index
]["text"]

ai_text = test_df.iloc[
    ai_index
]["text"]


print(
    "\n============================================================"
)

print(
    "SELECTED HUMAN SAMPLE"
)

print(
    "============================================================"
)

print(
    human_text[:500]
)


print(
    "\n============================================================"
)

print(
    "SELECTED AI SAMPLE"
)

print(
    "============================================================"
)

print(
    ai_text[:500]
)


# ============================================================
# 28. FEATURE SHAP FOR HUMAN SAMPLE
# ============================================================

human_feature_shap, human_feature_importance = (
    explain_syntactic_features(
        human_text,
        sample_name="Human_Sample",
        background_size=SHAP_BACKGROUND_SIZE,
        nsamples=SHAP_NSAMPLES,
    )
)


# ============================================================
# 29. FEATURE SHAP FOR AI SAMPLE
# ============================================================

ai_feature_shap, ai_feature_importance = (
    explain_syntactic_features(
        ai_text,
        sample_name="AI_Sample",
        background_size=SHAP_BACKGROUND_SIZE,
        nsamples=SHAP_NSAMPLES,
    )
)


# ============================================================
# 30. OPTIONAL NORMAL PREDICTION FUNCTION
# ============================================================

def test_sentence(sentence):

    (
        sequences,
        syntactic_scaled,
        cleaned_texts,
        original_features,
    ) = prepare_model_input(
        [sentence]
    )

    prediction = model.predict(
        [
            sequences,
            syntactic_scaled,
        ],
        verbose=0,
    )[0][0]

    predicted_label = (
        "AI-generated"
        if prediction >= 0.5
        else "Human-written"
    )

    print(
        "\n============================================================"
    )

    print(
        "MFAD PREDICTION"
    )

    print(
        "============================================================"
    )

    print(
        "Prediction:",
        predicted_label,
    )

    print(
        "AI-generated probability:",
        f"{prediction:.4f}",
    )

    print(
        "Human-written probability:",
        f"{1 - prediction:.4f}",
    )

    print(
        "\nHandcrafted Features:"
    )

    for name, value in zip(
        FEATURE_NAMES,
        original_features[0],
    ):
        print(
            f"{name}: {value:.4f}"
        )


# ============================================================
# 31. FINISHED
# ============================================================

print(
    "\n============================================================"
)

print(
    "ALL MFAD EXPERIMENTS COMPLETED SUCCESSFULLY."
)

print(
    "Feature-level SHAP only was used."
)

print(
    "No text/token-level SHAP was executed."
)

print(
    "============================================================"
)
