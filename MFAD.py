# ============================================================
# MFAD: Multi-Feature Accurate Detection
# CNN + BiLSTM + 17 Handcrafted Features
#
# 17 Handcrafted Features:
# 1.  Total Words
# 2.  Total Sentences
# 3.  Total Unique Words
# 4.  Type-Token Ratio
# 5.  Total Stop Words
# 6.  Total Punctuation
# 7.  Total Discourse Markers
# 8.  Total Spelling Errors
# 9.  Total Grammar Errors
# 10. Readability Score
# 11. Syllable Count
# 12. Average Sentence Length
# 13. Average Word Length
# 14. POS Tag Distribution
# 15. Sentence Complexity
# 16. Top Bigram Frequency
# 17. Top Trigram Frequency
# ============================================================


# ============================================================
# 1. IMPORT LIBRARIES
# ============================================================

import json
import os
import re
import string
import warnings
from collections import Counter

import numpy as np
import pandas as pd

import nltk
import spacy
import textstat

from nltk import word_tokenize, pos_tag, sent_tokenize
from nltk.corpus import stopwords

from spellchecker import SpellChecker
import language_tool_python

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report
)

from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import (
    Input,
    Embedding,
    Conv1D,
    MaxPooling1D,
    Bidirectional,
    LSTM,
    Dense,
    Concatenate,
    Dropout
)

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences


warnings.filterwarnings("ignore")


# ============================================================
# 2. NLTK RESOURCES
# ============================================================

print("Checking NLTK resources...")

nltk.download("punkt")
nltk.download("punkt_tab")
nltk.download("stopwords")
nltk.download("averaged_perceptron_tagger")
nltk.download("averaged_perceptron_tagger_eng")
nltk.download("wordnet")


# ============================================================
# 3. LOAD NLP MODELS
# ============================================================

print("Loading spaCy model...")

nlp = spacy.load("en_core_web_sm")

stop_words = set(stopwords.words("english"))

spell_checker = SpellChecker(language="en")

print("Loading LanguageTool...")

# LanguageTool is used for grammar-error detection.
# It can be relatively slow, especially for large datasets.
grammar_tool = language_tool_python.LanguageTool("en-US")


# ============================================================
# 4. FEATURE NAMES
# ============================================================

FEATURE_NAMES = [
    "Total Words",
    "Total Sentences",
    "Total Unique Words",
    "Type-Token Ratio",
    "Total Stop Words",
    "Total Punctuation",
    "Total Discourse Markers",
    "Total Spelling Errors",
    "Total Grammar Errors",
    "Readability Score",
    "Syllable Count",
    "Average Sentence Length",
    "Average Word Length",
    "POS Tag Distribution",
    "Sentence Complexity",
    "Top Bigram Frequency",
    "Top Trigram Frequency",
]

print("\nNumber of handcrafted features:", len(FEATURE_NAMES))

assert len(FEATURE_NAMES) == 17, \
    "FEATURE_NAMES must contain exactly 17 features."


# ============================================================
# 5. DISCOURSE MARKERS
# ============================================================

DISCOURSE_MARKERS = [
    "however",
    "therefore",
    "moreover",
    "furthermore",
    "nevertheless",
    "nonetheless",
    "consequently",
    "thus",
    "hence",
    "additionally",
    "meanwhile",
    "otherwise",
    "instead",
    "similarly",
    "likewise",
    "in contrast",
    "on the other hand",
    "for example",
    "for instance",
    "in addition",
    "as a result",
    "in conclusion",
    "to conclude",
    "firstly",
    "secondly",
    "finally",
    "in fact",
    "indeed",
    "specifically",
    "generally",
    "overall",
    "although",
    "because",
    "besides",
    "still",
]


# ============================================================
# 6. LOAD DATASET
# ============================================================

def load_hc3_json(path):

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    texts = []
    labels = []

    for entry in data:

        # ----------------------------------------------------
        # Human
        # ----------------------------------------------------

        for h in entry.get("Human", []):
            if isinstance(h, str) and h.strip():
                texts.append(h)
                labels.append(0)

        # ----------------------------------------------------
        # GPT-4
        # ----------------------------------------------------

        for a in entry.get("GPT-4", []):
            if isinstance(a, str) and a.strip():
                texts.append(a)
                labels.append(1)

    df = pd.DataFrame({
        "text": texts,
        "label": labels
    })

    return df


# ============================================================
# 7. DATASET PATH
# ============================================================

df = load_hc3_json(
    "HumanVsGPT-4.json"
)

# Other possible datasets:
#
# df = load_hc3_json("IEEE-ChatGPT-PolishWithHuman.json")
# df = load_hc3_json("IEEE-ChatGPT-GenerationWithHuman.json")
# df = load_hc3_json("HumanVsYi-Large.json")
# df = load_hc3_json("HumanVsMistral.json")
# df = load_hc3_json("HumanVsgGemma.json")
# df = load_hc3_json("HumanVsllama.json")
# df = load_hc3_json("HumanVsQwene.json")


print("\nDataset size:", len(df))
print("\nClass distribution:")
print(df["label"].value_counts())


# ============================================================
# 8. TEXT CLEANING FOR CNN/BiLSTM
# ============================================================

def clean_text(text):

    if not isinstance(text, str):
        return ""

    text = str(text)

    # --------------------------------------------------------
    # Remove URLs
    # --------------------------------------------------------

    text = re.sub(
        r"http\S+|www\S+|https\S+",
        " ",
        text,
        flags=re.IGNORECASE
    )

    # --------------------------------------------------------
    # Remove hashtags
    # --------------------------------------------------------

    text = re.sub(
        r"#\w+",
        " ",
        text
    )

    # --------------------------------------------------------
    # Remove currency symbols and units
    # --------------------------------------------------------

    text = re.sub(
        r"[\$\€\£\₹\¥\₽\₩\¢]+[\d,.]*"
        r"|\d+[\s]?(USD|EUR|EGP|GBP|JPY|INR)",
        " ",
        text,
        flags=re.IGNORECASE
    )

    # --------------------------------------------------------
    # Lowercase
    # --------------------------------------------------------

    text = text.lower()

    # --------------------------------------------------------
    # Keep alphabetic characters and whitespace
    # --------------------------------------------------------

    text = re.sub(
        r"[^a-z\s]",
        " ",
        text
    )

    # --------------------------------------------------------
    # Normalize spaces
    # --------------------------------------------------------

    text = re.sub(
        r"\s+",
        " ",
        text
    ).strip()

    return text


df["text_clean"] = df["text"].apply(clean_text)


# ============================================================
# 9. HELPER FUNCTIONS
# ============================================================

def safe_word_tokenize(text):
    """
    Safely tokenize words.
    """

    try:
        return word_tokenize(text)
    except Exception:
        return re.findall(r"\b[a-zA-Z]+\b", text)


def safe_sentence_tokenize(text):
    """
    Safely tokenize sentences.
    """

    try:
        return sent_tokenize(text)
    except Exception:

        sentences = re.split(
            r"[.!?]+",
            text
        )

        return [
            s.strip()
            for s in sentences
            if s.strip()
        ]


# ============================================================
# 10. TOTAL DISCOURSE MARKERS
# ============================================================

def count_discourse_markers(text):

    text_lower = text.lower()

    total = 0

    for marker in DISCOURSE_MARKERS:

        pattern = r"\b" + re.escape(marker) + r"\b"

        matches = re.findall(
            pattern,
            text_lower
        )

        total += len(matches)

    return total


# ============================================================
# 11. SPELLING ERRORS
# ============================================================

def count_spelling_errors(tokens):

    if not tokens:
        return 0

    # Keep alphabetic words only
    words = [
        word.lower()
        for word in tokens
        if re.fullmatch(r"[a-zA-Z]+", word)
    ]

    if not words:
        return 0

    # Remove very short words because spell-checking
    # "a", "I", etc. can introduce noise.
    words_for_checking = [
        word
        for word in words
        if len(word) > 2
    ]

    if not words_for_checking:
        return 0

    misspelled = spell_checker.unknown(
        words_for_checking
    )

    return len(misspelled)


# ============================================================
# 12. GRAMMAR ERRORS
# ============================================================

def count_grammar_errors(text):

    if not text.strip():
        return 0

    try:

        matches = grammar_tool.check(text)

        return len(matches)

    except Exception as ex:

        print(
            "Grammar checking error:",
            ex
        )

        return 0


# ============================================================
# 13. POS TAG DISTRIBUTION
# ============================================================

def calculate_pos_distribution(pos_tags):

    if not pos_tags:
        return 0.0

    # --------------------------------------------------------
    # Count POS tags
    # --------------------------------------------------------

    pos_counter = Counter(
        tag
        for _, tag in pos_tags
    )

    total = sum(
        pos_counter.values()
    )

    if total == 0:
        return 0.0

    # --------------------------------------------------------
    # Calculate normalized POS entropy
    #
    # H = -sum(p * log2(p))
    #
    # Higher value:
    # more diverse POS distribution
    #
    # Lower value:
    # more concentrated POS distribution
    # --------------------------------------------------------

    probabilities = [
        count / total
        for count in pos_counter.values()
    ]

    entropy = -sum(
        p * np.log2(p)
        for p in probabilities
        if p > 0
    )

    return float(entropy)


# ============================================================
# 14. SENTENCE COMPLEXITY
# ============================================================

def calculate_sentence_complexity(text):

    if not text.strip():
        return 0.0

    try:

        doc = nlp(text)

        sentence_depths = []

        for sent in doc.sents:

            depths = []

            for token in sent:

                depth = 0

                current = token

                visited = set()

                while (
                    current.head != current
                    and current.i not in visited
                ):

                    visited.add(current.i)

                    depth += 1

                    current = current.head

                depths.append(depth)

            if depths:

                sentence_depths.append(
                    np.mean(depths)
                )

        if not sentence_depths:
            return 0.0

        return float(
            np.mean(sentence_depths)
        )

    except Exception as ex:

        print(
            "Sentence complexity error:",
            ex
        )

        return 0.0


# ============================================================
# 15. BIGRAM FREQUENCY
# ============================================================

def calculate_top_ngram_frequency(
    tokens,
    n
):

    if len(tokens) < n:
        return 0.0

    ngrams = [
        tuple(tokens[i:i + n])
        for i in range(len(tokens) - n + 1)
    ]

    if not ngrams:
        return 0.0

    counts = Counter(ngrams)

    return float(
        max(counts.values())
    )


# ============================================================
# 16. EXTRACT 17 FEATURES
# ============================================================

def extract_features(text):

    # --------------------------------------------------------
    # Make sure input is string
    # --------------------------------------------------------

    if not isinstance(text, str):
        text = ""

    original_text = text

    # --------------------------------------------------------
    # Tokenization
    # --------------------------------------------------------

    tokens = safe_word_tokenize(
        original_text
    )

    # Only alphabetic words
    word_tokens = [
        token.lower()
        for token in tokens
        if re.fullmatch(
            r"[a-zA-Z]+",
            token
        )
    ]

    # --------------------------------------------------------
    # Sentences
    # --------------------------------------------------------

    sentences = safe_sentence_tokenize(
        original_text
    )

    # Remove empty sentences
    sentences = [
        sentence
        for sentence in sentences
        if sentence.strip()
    ]

    # --------------------------------------------------------
    # Basic statistics
    # --------------------------------------------------------

    total_words = len(word_tokens)

    total_sentences = len(sentences)

    total_unique_words = len(
        set(word_tokens)
    )

    # --------------------------------------------------------
    # Type Token Ratio
    # --------------------------------------------------------

    if total_words > 0:

        type_token_ratio = (
            total_unique_words /
            total_words
        )

    else:

        type_token_ratio = 0.0

    # --------------------------------------------------------
    # Stop words
    # --------------------------------------------------------

    total_stop_words = sum(
        1
        for word in word_tokens
        if word in stop_words
    )

    # --------------------------------------------------------
    # Punctuation
    #
    # IMPORTANT:
    # We calculate this from ORIGINAL text,
    # not cleaned text.
    # --------------------------------------------------------

    total_punctuation = sum(
        1
        for char in original_text
        if char in string.punctuation
    )

    # --------------------------------------------------------
    # Discourse markers
    # --------------------------------------------------------

    total_discourse_markers = (
        count_discourse_markers(
            original_text
        )
    )

    # --------------------------------------------------------
    # Spelling errors
    # --------------------------------------------------------

    total_spelling_errors = (
        count_spelling_errors(
            word_tokens
        )
    )

    # --------------------------------------------------------
    # Grammar errors
    # --------------------------------------------------------

    total_grammar_errors = (
        count_grammar_errors(
            original_text
        )
    )

    # --------------------------------------------------------
    # Readability
    # --------------------------------------------------------

    try:

        readability_score = (
            textstat.flesch_reading_ease(
                original_text
            )
        )

        if not np.isfinite(
            readability_score
        ):
            readability_score = 0.0

    except Exception:

        readability_score = 0.0

    # --------------------------------------------------------
    # Syllable count
    # --------------------------------------------------------

    try:

        syllable_count = (
            textstat.syllable_count(
                original_text
            )
        )

    except Exception:

        syllable_count = 0

    # --------------------------------------------------------
    # Average sentence length
    # --------------------------------------------------------

    if total_sentences > 0:

        average_sentence_length = (
            total_words /
            total_sentences
        )

    else:

        average_sentence_length = 0.0

    # --------------------------------------------------------
    # Average word length
    # --------------------------------------------------------

    if total_words > 0:

        average_word_length = (
            np.mean(
                [
                    len(word)
                    for word in word_tokens
                ]
            )
        )

    else:

        average_word_length = 0.0

    # --------------------------------------------------------
    # POS tags
    # --------------------------------------------------------

    try:

        pos_tags = pos_tag(
            word_tokens
        )

    except Exception:

        pos_tags = []

    # --------------------------------------------------------
    # POS Tag Distribution
    #
    # We represent the distribution using
    # POS entropy as one scalar feature.
    # --------------------------------------------------------

    pos_tag_distribution = (
        calculate_pos_distribution(
            pos_tags
        )
    )

    # --------------------------------------------------------
    # Sentence Complexity
    #
    # Average dependency-tree depth
    # --------------------------------------------------------

    sentence_complexity = (
        calculate_sentence_complexity(
            original_text
        )
    )

    # --------------------------------------------------------
    # Top Bigram Frequency
    # --------------------------------------------------------

    top_bigram_frequency = (
        calculate_top_ngram_frequency(
            word_tokens,
            2
        )
    )

    # --------------------------------------------------------
    # Top Trigram Frequency
    # --------------------------------------------------------

    top_trigram_frequency = (
        calculate_top_ngram_frequency(
            word_tokens,
            3
        )
    )

    # ========================================================
    # FINAL 17 FEATURES
    # ========================================================

    features = [

        # 1
        total_words,

        # 2
        total_sentences,

        # 3
        total_unique_words,

        # 4
        type_token_ratio,

        # 5
        total_stop_words,

        # 6
        total_punctuation,

        # 7
        total_discourse_markers,

        # 8
        total_spelling_errors,

        # 9
        total_grammar_errors,

        # 10
        readability_score,

        # 11
        syllable_count,

        # 12
        average_sentence_length,

        # 13
        average_word_length,

        # 14
        pos_tag_distribution,

        # 15
        sentence_complexity,

        # 16
        top_bigram_frequency,

        # 17
        top_trigram_frequency,
    ]

    # --------------------------------------------------------
    # Verify exactly 17 features
    # --------------------------------------------------------

    assert len(features) == 17, (
        f"Expected 17 features, "
        f"but extracted {len(features)}"
    )

    return features


# ============================================================
# 17. EXTRACT HANDCRAFTED FEATURES FOR DATASET
# ============================================================

print("\nExtracting 17 handcrafted features...")

X_syntactic = np.array(
    [
        extract_features(text)
        for text in df["text"]
    ],
    dtype=np.float32
)


# ============================================================
# 18. CREATE FEATURE DATAFRAME
# ============================================================

features_df = pd.DataFrame(
    X_syntactic,
    columns=FEATURE_NAMES
)

print("\n============================================================")
print("17 HANDCRAFTED FEATURES")
print("============================================================")

print(
    features_df.head()
)

print(
    "\nFeature matrix shape:",
    X_syntactic.shape
)


# ============================================================
# 19. VERIFY FEATURE COUNT
# ============================================================

print(
    "\nNumber of handcrafted features:",
    X_syntactic.shape[1]
)

assert X_syntactic.shape[1] == 17


# ============================================================
# 20. CHECK FOR NAN / INFINITE VALUES
# ============================================================

X_syntactic = np.nan_to_num(
    X_syntactic,
    nan=0.0,
    posinf=0.0,
    neginf=0.0
)


# ============================================================
# 21. DISPLAY FEATURE STATISTICS
# ============================================================

print("\n============================================================")
print("FEATURE STATISTICS")
print("============================================================")

for i, feature_name in enumerate(
    FEATURE_NAMES
):

    print(
        f"{i + 1:02d}. "
        f"{feature_name:<30} "
        f"Mean = {X_syntactic[:, i].mean():.4f} "
        f"Std = {X_syntactic[:, i].std():.4f}"
    )


# ============================================================
# 22. SAVE FEATURE DATASET
# ============================================================

features_output = df[
    ["text", "label"]
].copy()

for i, feature_name in enumerate(
    FEATURE_NAMES
):

    features_output[
        feature_name
    ] = X_syntactic[:, i]


features_output.to_csv(
    "MFAD_17_Handcrafted_Features.csv",
    index=False,
    encoding="utf-8-sig"
)

print(
    "\n17 handcrafted features saved to:"
    " MFAD_17_Handcrafted_Features.csv"
)


# ============================================================
# 23. CNN/BILSTM TEXT SEQUENCES
# ============================================================

MAX_SEQUENCE_LENGTH = 300

tokenizer = Tokenizer(
    oov_token="<OOV>"
)

tokenizer.fit_on_texts(
    df["text_clean"]
)

X_seq = pad_sequences(
    tokenizer.texts_to_sequences(
        df["text_clean"]
    ),
    maxlen=MAX_SEQUENCE_LENGTH,
    padding="post",
    truncating="post"
)

word_index = tokenizer.word_index

print(
    "\nVocabulary size:",
    len(word_index)
)

print(
    "Sequence matrix shape:",
    X_seq.shape
)


# ============================================================
# 24. LOAD GLOVE
# ============================================================

GLOVE_PATH = (
    r"G:\Code\glove.6B.100d.txt"
)

embedding_dim = 100

embedding_index = {}

print(
    "\nLoading GloVe embeddings..."
)

with open(
    GLOVE_PATH,
    encoding="utf-8"
) as f:

    for line in f:

        values = line.rstrip().split()

        word = values[0]

        vector = np.asarray(
            values[1:],
            dtype="float32"
        )

        embedding_index[
            word
        ] = vector


print(
    "GloVe vocabulary:",
    len(embedding_index)
)


# ============================================================
# 25. CREATE EMBEDDING MATRIX
# ============================================================

embedding_matrix = np.zeros(
    (
        len(word_index) + 1,
        embedding_dim
    ),
    dtype="float32"
)

found_embeddings = 0

for word, index in word_index.items():

    vector = embedding_index.get(word)

    if vector is not None:

        embedding_matrix[
            index
        ] = vector

        found_embeddings += 1


print(
    "Words with GloVe embeddings:",
    found_embeddings
)

print(
    "Embedding matrix shape:",
    embedding_matrix.shape
)


# ============================================================
# 26. TRAIN / TEST SPLIT
# ============================================================

y = df["label"].values

(
    X_train_seq,
    X_test_seq,
    X_train_syn,
    X_test_syn,
    y_train,
    y_test
) = train_test_split(

    X_seq,
    X_syntactic,
    y,

    test_size=0.20,

    random_state=42,

    stratify=y
)


print("\n============================================================")
print("TRAIN / TEST")
print("============================================================")

print(
    "Training sequences:",
    X_train_seq.shape
)

print(
    "Testing sequences:",
    X_test_seq.shape
)

print(
    "Training handcrafted:",
    X_train_syn.shape
)

print(
    "Testing handcrafted:",
    X_test_syn.shape
)


# ============================================================
# 27. SCALE 17 HANDCRAFTED FEATURES
# ============================================================

# Scaling is recommended because the 17 features
# have very different numerical ranges.

scaler = StandardScaler()

X_train_syn = scaler.fit_transform(
    X_train_syn
)

X_test_syn = scaler.transform(
    X_test_syn
)


print(
    "\nScaled handcrafted feature shape:",
    X_train_syn.shape
)


# ============================================================
# 28. BUILD MFAD MODEL
# ============================================================

def build_model():

    # --------------------------------------------------------
    # Text input
    # --------------------------------------------------------

    text_input = Input(
        shape=(MAX_SEQUENCE_LENGTH,),
        name="text_input"
    )

    # --------------------------------------------------------
    # 17 handcrafted feature input
    # --------------------------------------------------------

    meta_input = Input(
        shape=(17,),
        name="handcrafted_features"
    )

    # --------------------------------------------------------
    # GloVe embedding
    # --------------------------------------------------------

    embedding = Embedding(
        input_dim=len(word_index) + 1,

        output_dim=embedding_dim,

        weights=[embedding_matrix],

        input_length=MAX_SEQUENCE_LENGTH,

        trainable=False,

        name="glove_embedding"
    )(text_input)

    # --------------------------------------------------------
    # CNN
    # --------------------------------------------------------

    conv = Conv1D(
        filters=128,
        kernel_size=5,
        activation="relu",
        name="cnn"
    )(embedding)

    # --------------------------------------------------------
    # Max Pooling
    # --------------------------------------------------------

    pool = MaxPooling1D(
        pool_size=2,
        name="max_pooling"
    )(conv)

    # --------------------------------------------------------
    # BiLSTM
    # --------------------------------------------------------

    bilstm = Bidirectional(
        LSTM(
            64
        ),
        name="bilstm"
    )(pool)

    # --------------------------------------------------------
    # Concatenate semantic + handcrafted features
    # --------------------------------------------------------

    merged = Concatenate(
        name="concat_features"
    )(
        [
            bilstm,
            meta_input
        ]
    )

    # --------------------------------------------------------
    # Dense
    # --------------------------------------------------------

    dense = Dense(
        64,
        activation="relu",
        name="dense"
    )(merged)

    # --------------------------------------------------------
    # Dropout
    # --------------------------------------------------------

    dropout = Dropout(
        0.5,
        name="dropout"
    )(dense)

    # --------------------------------------------------------
    # Output
    # --------------------------------------------------------

    output = Dense(
        1,
        activation="sigmoid",
        name="output"
    )(dropout)

    # --------------------------------------------------------
    # Model
    # --------------------------------------------------------

    model = Model(
        inputs=[
            text_input,
            meta_input
        ],

        outputs=output,

        name="MFAD"
    )

    # --------------------------------------------------------
    # Compile
    # --------------------------------------------------------

    model.compile(

        loss="binary_crossentropy",

        optimizer="adam",

        metrics=[
            "accuracy"
        ]
    )

    return model


# ============================================================
# 29. CREATE MODEL
# ============================================================

model = build_model()


print("\n============================================================")
print("MFAD MODEL")
print("============================================================")

model.summary()


# ============================================================
# 30. TRAIN MODEL
# ============================================================

history = model.fit(

    [
        X_train_seq,
        X_train_syn
    ],

    y_train,

    epochs=5,

    batch_size=32,

    validation_split=0.10,

    verbose=1
)


# ============================================================
# 31. SAVE MODEL
# ============================================================

MODEL_PATH = (
    r"G:\Code\Casestudy1"
    r"\human_ai_classifier2.keras"
)

model.save(
    MODEL_PATH
)

print(
    "\nModel saved to:",
    MODEL_PATH
)


# ============================================================
# 32. PREDICTION
# ============================================================

y_prob = model.predict(
    [
        X_test_seq,
        X_test_syn
    ],
    verbose=0
).ravel()


# Threshold = 0.5
y_pred = (
    y_prob >= 0.5
).astype(int)


# ============================================================
# 33. EVALUATION
# ============================================================

accuracy = accuracy_score(
    y_test,
    y_pred
)

precision = precision_score(
    y_test,
    y_pred,
    zero_division=0
)

recall = recall_score(
    y_test,
    y_pred,
    zero_division=0
)

f1 = f1_score(
    y_test,
    y_pred,
    zero_division=0
)

# IMPORTANT:
# AUROC should use prediction probabilities,
# NOT binary predictions.

auroc = roc_auc_score(
    y_test,
    y_prob
)


print("\n============================================================")
print("MFAD RESULTS")
print("============================================================")

print(
    f"Accuracy : {accuracy:.4f}"
)

print(
    f"Precision: {precision:.4f}"
)

print(
    f"Recall   : {recall:.4f}"
)

print(
    f"F1 Score : {f1:.4f}"
)

print(
    f"AUROC    : {auroc:.4f}"
)


# ============================================================
# 34. CLASSIFICATION REPORT
# ============================================================

print(
    "\nClassification Report:\n"
)

print(
    classification_report(
        y_test,
        y_pred,
        target_names=[
            "Human",
            "AI-generated"
        ],
        zero_division=0
    )
)


# ============================================================
# 35. CONFUSION MATRIX
# ============================================================

cm = confusion_matrix(
    y_test,
    y_pred
)

print(
    "Confusion Matrix:"
)

print(cm)


# ============================================================
# 36. FALSE POSITIVE RATE
# ============================================================

tn, fp, fn, tp = cm.ravel()

fpr = (
    fp / (fp + tn)
    if (fp + tn) > 0
    else 0.0
)

print(
    "\nFalse Positive Rate (FPR):",
    f"{fpr:.4f}"
)


# ============================================================
# 37. PRINT FEATURE VALUES FOR ONE TEXT
# ============================================================

def print_feature_values(
    text
):

    features = extract_features(
        text
    )

    print(
        "\n============================================================"
    )

    print(
        "17 HANDCRAFTED FEATURE VALUES"
    )

    print(
        "============================================================"
    )

    for i, (
        name,
        value
    ) in enumerate(
        zip(
            FEATURE_NAMES,
            features
        ),
        start=1
    ):

        print(
            f"{i:02d}. "
            f"{name:<30}: "
            f"{value:.6f}"
            if isinstance(
                value,
                float
            )
            else
            f"{i:02d}. "
            f"{name:<30}: "
            f"{value}"
        )


# ============================================================
# 38. TEST SENTENCE
# ============================================================

def test_sentence(
    sentence
):

    # --------------------------------------------------------
    # Clean text for neural network
    # --------------------------------------------------------

    clean = clean_text(
        sentence
    )

    # --------------------------------------------------------
    # Convert to sequence
    # --------------------------------------------------------

    seq = pad_sequences(

        tokenizer.texts_to_sequences(
            [clean]
        ),

        maxlen=MAX_SEQUENCE_LENGTH,

        padding="post",

        truncating="post"
    )

    # --------------------------------------------------------
    # Extract exactly 17 handcrafted features
    # --------------------------------------------------------

    syn_raw = np.array(
        [
            extract_features(
                sentence
            )
        ],
        dtype=np.float32
    )

    # --------------------------------------------------------
    # Verify shape
    # --------------------------------------------------------

    assert syn_raw.shape == (
        1,
        17
    )

    # --------------------------------------------------------
    # Scale using training scaler
    # --------------------------------------------------------

    syn = scaler.transform(
        syn_raw
    )

    # --------------------------------------------------------
    # Prediction
    # --------------------------------------------------------

    prediction = model.predict(
        [
            seq,
            syn
        ],
        verbose=0
    )[0][0]

    # --------------------------------------------------------
    # Semantic representation
    #
    # concat_features receives:
    # [BiLSTM features, handcrafted features]
    #
    # We extract the BiLSTM output.
    # --------------------------------------------------------

    semantic_model = Model(
        inputs=model.inputs,

        outputs=model.get_layer(
            "bilstm"
        ).output
    )

    semantic_features = (
        semantic_model.predict(
            [
                seq,
                syn
            ],
            verbose=0
        )
    )

    # --------------------------------------------------------
    # Concatenated vector
    # --------------------------------------------------------

    concatenated = np.concatenate(
        [
            semantic_features[0],
            syn[0]
        ]
    )

    # ========================================================
    # PRINT RESULT
    # ========================================================

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
        f"Prediction: "
        f"{'AI-generated' if prediction >= 0.5 else 'Human-written'}"
    )

    print(
        f"AI probability: {prediction:.4f}"
    )

    print(
        f"Human probability: {1 - prediction:.4f}"
    )

    # --------------------------------------------------------
    # 17 raw handcrafted features
    # --------------------------------------------------------

    print(
        "\n17 Handcrafted Features:"
    )

    for i, (
        name,
        value
    ) in enumerate(
        zip(
            FEATURE_NAMES,
            syn_raw[0]
        ),
        start=1
    ):

        print(
            f"{i:02d}. "
            f"{name:<30}: "
            f"{value:.6f}"
        )

    # --------------------------------------------------------
    # Semantic features
    # --------------------------------------------------------

    print(
        "\nSemantic Features:"
    )

    print(
        "Semantic vector shape:",
        semantic_features.shape
    )

    print(
        "Semantic vector length:",
        len(
            semantic_features[0]
        )
    )

    print(
        "First 10 semantic features:",
        semantic_features[0][:10]
    )

    # --------------------------------------------------------
    # Handcrafted scaled features
    # --------------------------------------------------------

    print(
        "\nScaled Handcrafted Features:"
    )

    print(
        syn[0]
    )

    # --------------------------------------------------------
    # Concatenated features
    # --------------------------------------------------------

    print(
        "\nConcatenated Features:"
    )

    print(
        "Concatenated vector length:",
        len(concatenated)
    )

    print(
        "Expected length:",
        len(
            semantic_features[0]
        ) + 17
    )

    print(
        "First 10 concatenated features:",
        concatenated[:10]
    )

    return {
        "prediction": prediction,
        "label": (
            "AI-generated"
            if prediction >= 0.5
            else "Human-written"
        ),
        "handcrafted_features": syn_raw[0],
        "semantic_features": semantic_features[0],
        "concatenated_features": concatenated
    }


# ============================================================
# 39. EXAMPLE TEST
# ============================================================

test_text = """
It sounds like you may be struggling with a condition called intrusive thoughts.
Intrusive thoughts are unwanted, involuntary thoughts, images, or urges that can
be distressing and can interfere with daily life. They can take many forms and
can be about a wide range of topics, including things that are disturbing or
inappropriate.

If you are struggling with intrusive thoughts and they are causing you distress
or interfering with your daily life, it may be helpful to seek treatment from a
mental health professional. A therapist or counselor can help you learn coping
strategies to manage your thoughts and improve your overall well-being.

Here are a few things you can try to help manage your intrusive thoughts:

Practice mindfulness: This involves paying attention to the present moment and
accepting your thoughts and feelings without judgment.

Use distraction techniques: Engaging in activities that take your mind off of
your thoughts can be helpful in managing them.

Challenge your thoughts: Try to look at your thoughts objectively and consider
whether they are accurate or not.

Seek support: Talking to a trusted friend or family member about your thoughts
can be helpful in managing them. You can also seek support from a mental health
professional.

It's important to remember that having intrusive thoughts does not mean you are
abnormal or that there is something wrong with you. Many people experience
intrusive thoughts from time to time, and with the right support and treatment,
you can learn to manage them and improve your overall well-being.
"""


# ============================================================
# 40. DISPLAY 17 FEATURES
# ============================================================

print_feature_values(
    test_text
)


# ============================================================
# 41. RUN PREDICTION
# ============================================================

result = test_sentence(
    test_text
)