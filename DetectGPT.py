# ============================================================
# DetectGPT Baseline
# Zero-Shot Machine-Generated Text Detection
#
# Dataset:
#     HumanVsGPT-4.json
#
# Labels:
#     Human = 0
#     GPT-4 = 1
#
# Method:
#     DetectGPT probability-curvature / perturbation method
#
# Scoring Model:
#     GPT-2 XL
#
# Perturbation Model:
#     T5-large
#
# ============================================================


# ============================================================
# 1. IMPORT LIBRARIES
# ============================================================

import os
import json
import re
import random
import warnings

import numpy as np
import pandas as pd
import torch

from tqdm import tqdm

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    AutoModelForSeq2SeqLM
)

from sklearn.model_selection import train_test_split

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
    roc_curve
)

warnings.filterwarnings("ignore")


# ============================================================
# 2. CONFIGURATION
# ============================================================

DATASET_PATH = "HumanVsGPT-4.json"

# ------------------------------------------------------------
# Scoring model
# ------------------------------------------------------------

SCORING_MODEL_NAME = "gpt2-xl"

# ------------------------------------------------------------
# Mask filling / perturbation model
# ------------------------------------------------------------

MASK_FILLING_MODEL_NAME = "t5-large"

# ------------------------------------------------------------
# Number of perturbations
# ------------------------------------------------------------

N_PERTURBATIONS = 100

# ------------------------------------------------------------
# Random seed
# ------------------------------------------------------------

RANDOM_SEED = 42

# ------------------------------------------------------------
# Test size
# ------------------------------------------------------------

TEST_SIZE = 0.20

# ------------------------------------------------------------
# Device
# ------------------------------------------------------------

DEVICE = (
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

print("Device:", DEVICE)


# ============================================================
# 3. RANDOM SEEDS
# ============================================================

random.seed(RANDOM_SEED)

np.random.seed(
    RANDOM_SEED
)

torch.manual_seed(
    RANDOM_SEED
)

if torch.cuda.is_available():

    torch.cuda.manual_seed_all(
        RANDOM_SEED
    )


# ============================================================
# 4. LOAD DATASET
# ============================================================

def load_hc3_json(
    path
):

    with open(
        path,
        "r",
        encoding="utf-8"
    ) as f:

        data = json.load(f)

    texts = []
    labels = []

    for entry in data:

        # ----------------------------------------------------
        # Human
        # ----------------------------------------------------

        for text in entry.get(
            "Human",
            []
        ):

            if (
                isinstance(text, str)
                and text.strip()
            ):

                texts.append(text)

                labels.append(0)

        # ----------------------------------------------------
        # GPT-4
        # ----------------------------------------------------

        for text in entry.get(
            "GPT-4",
            []
        ):

            if (
                isinstance(text, str)
                and text.strip()
            ):

                texts.append(text)

                labels.append(1)

    return pd.DataFrame(
        {
            "text": texts,
            "label": labels
        }
    )


# ============================================================
# 5. LOAD DATA
# ============================================================

df = load_hc3_json(
    DATASET_PATH
)

print(
    "\nDataset size:",
    len(df)
)

print(
    "\nClass distribution:"
)

print(
    df["label"].value_counts()
)


# ============================================================
# 6. TRAIN / TEST SPLIT
# ============================================================

df_train, df_test = train_test_split(

    df,

    test_size=TEST_SIZE,

    random_state=RANDOM_SEED,

    stratify=df["label"]
)


print(
    "\nTraining samples:",
    len(df_train)
)

print(
    "Testing samples:",
    len(df_test)
)


# ============================================================
# 7. LOAD SCORING MODEL
# ============================================================

print(
    "\nLoading scoring tokenizer..."
)

scoring_tokenizer = AutoTokenizer.from_pretrained(
    SCORING_MODEL_NAME
)

if scoring_tokenizer.pad_token is None:

    scoring_tokenizer.pad_token = (
        scoring_tokenizer.eos_token
    )


print(
    "Loading scoring model..."
)

scoring_model = AutoModelForCausalLM.from_pretrained(
    SCORING_MODEL_NAME
)

scoring_model.to(
    DEVICE
)

scoring_model.eval()


# ============================================================
# 8. LOAD MASK-FILLING MODEL
# ============================================================

print(
    "\nLoading T5 tokenizer..."
)

mask_tokenizer = AutoTokenizer.from_pretrained(
    MASK_FILLING_MODEL_NAME
)

print(
    "Loading T5 model..."
)

mask_model = AutoModelForSeq2SeqLM.from_pretrained(
    MASK_FILLING_MODEL_NAME
)

mask_model.to(
    DEVICE
)

mask_model.eval()


# ============================================================
# 9. COMPUTE LOG-LIKELIHOOD
# ============================================================

def get_log_likelihood(
    text
):

    if not text.strip():
        return 0.0

    tokens = scoring_tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=1024
    )

    input_ids = tokens[
        "input_ids"
    ].to(DEVICE)

    attention_mask = tokens[
        "attention_mask"
    ].to(DEVICE)

    if input_ids.shape[1] < 2:

        return 0.0

    with torch.no_grad():

        outputs = scoring_model(
            input_ids=input_ids,
            attention_mask=attention_mask
        )

        logits = outputs.logits

    # --------------------------------------------------------
    # Shift logits and labels
    # --------------------------------------------------------

    shift_logits = logits[
        :,
        :-1,
        :
    ]

    shift_labels = input_ids[
        :,
        1:
    ]

    # --------------------------------------------------------
    # Log softmax
    # --------------------------------------------------------

    log_probs = torch.log_softmax(
        shift_logits,
        dim=-1
    )

    token_log_probs = (
        log_probs
        .gather(
            2,
            shift_labels.unsqueeze(-1)
        )
        .squeeze(-1)
    )

    # --------------------------------------------------------
    # Mean log probability
    # --------------------------------------------------------

    mean_log_prob = (
        token_log_probs
        .mean()
        .item()
    )

    return mean_log_prob


# ============================================================
# 10. SPLIT TEXT INTO CHUNKS FOR PERTURBATION
# ============================================================

def split_into_words(
    text
):

    return re.findall(
        r"\S+",
        text
    )


# ============================================================
# 11. CREATE MASKED TEXT
# ============================================================

def create_masked_text(
    text,
    mask_ratio=0.15
):

    words = split_into_words(
        text
    )

    if len(words) < 10:

        return None

    n_masks = max(
        1,
        int(
            len(words)
            * mask_ratio
        )
    )

    indices = random.sample(
        range(len(words)),
        min(
            n_masks,
            len(words)
        )
    )

    indices = sorted(
        indices
    )

    masked_words = words.copy()

    sentinel_id = 0

    previous_index = -1

    for index in indices:

        if index == previous_index:
            continue

        masked_words[index] = (
            f"<extra_id_{sentinel_id}>"
        )

        sentinel_id += 1

        previous_index = index

    return " ".join(
        masked_words
    )


# ============================================================
# 12. GENERATE T5 PERTURBATION
# ============================================================

def generate_perturbation(
    text
):

    masked_text = create_masked_text(
        text
    )

    if masked_text is None:

        return None

    inputs = mask_tokenizer(
        masked_text,
        return_tensors="pt",
        truncation=True,
        max_length=512
    )

    input_ids = inputs[
        "input_ids"
    ].to(DEVICE)

    attention_mask = inputs[
        "attention_mask"
    ].to(DEVICE)

    with torch.no_grad():

        outputs = mask_model.generate(

            input_ids=input_ids,

            attention_mask=attention_mask,

            max_length=512,

            do_sample=True,

            top_p=0.95,

            temperature=1.0,

            num_return_sequences=1
        )

    generated = (
        mask_tokenizer.decode(
            outputs[0],
            skip_special_tokens=False
        )
    )

    # --------------------------------------------------------
    # Extract generated text between sentinel tokens
    # --------------------------------------------------------

    generated = re.sub(
        r"<extra_id_\d+>",
        " ",
        generated
    )

    generated = re.sub(
        r"\s+",
        " ",
        generated
    ).strip()

    # --------------------------------------------------------
    # If reconstruction is invalid
    # --------------------------------------------------------

    if len(generated.split()) < 3:

        return None

    return generated


# ============================================================
# 13. DETECTGPT SCORE
# ============================================================

def detectgpt_score(
    text,
    n_perturbations=N_PERTURBATIONS
):

    # --------------------------------------------------------
    # Original likelihood
    # --------------------------------------------------------

    original_ll = (
        get_log_likelihood(
            text
        )
    )

    perturbation_scores = []

    for _ in range(
        n_perturbations
    ):

        perturbed = (
            generate_perturbation(
                text
            )
        )

        if (
            perturbed is None
            or len(perturbed.strip()) == 0
        ):
            continue

        perturbed_ll = (
            get_log_likelihood(
                perturbed
            )
        )

        perturbation_scores.append(
            perturbed_ll
        )

    if not perturbation_scores:

        return 0.0

    # --------------------------------------------------------
    # DetectGPT curvature criterion
    #
    # Higher:
    # original is relatively more likely
    # than perturbed versions.
    # --------------------------------------------------------

    mean_perturbed_ll = np.mean(
        perturbation_scores
    )

    score = (
        original_ll
        - mean_perturbed_ll
    )

    return float(score)


# ============================================================
# 14. RUN DETECTGPT ON DATASET
# ============================================================

scores = []
labels = []

print(
    "\n============================================================"
)

print(
    "Running DetectGPT"
)

print(
    "============================================================"
)

for index, row in tqdm(
    df_test.iterrows(),
    total=len(df_test)
):

    text = row["text"]

    label = row["label"]

    try:

        score = detectgpt_score(
            text,
            N_PERTURBATIONS
        )

    except Exception as ex:

        print(
            "\nError at sample",
            index,
            ":",
            ex
        )

        score = 0.0

    scores.append(
        score
    )

    labels.append(
        label
    )


# ============================================================
# 15. RESULTS DATAFRAME
# ============================================================

results = pd.DataFrame(
    {
        "text": df_test[
            "text"
        ].values,

        "label": labels,

        "DetectGPT_Score": scores
    }
)


# ============================================================
# 16. NORMALIZE SCORE TO PROBABILITY
# ============================================================

def sigmoid(
    x
):

    x = np.clip(
        x,
        -50,
        50
    )

    return (
        1.0
        /
        (
            1.0
            + np.exp(-x)
        )
    )


results[
    "DetectGPT_Probability"
] = results[
    "DetectGPT_Score"
].apply(
    sigmoid
)


# ============================================================
# 17. CLASSIFICATION
# ============================================================

y_true = results[
    "label"
].values

y_score = results[
    "DetectGPT_Score"
].values

y_probability = results[
    "DetectGPT_Probability"
].values


# Threshold
#
# IMPORTANT:
# For a rigorous paper comparison, the threshold should
# preferably be selected on a validation set rather than
# assumed to be universally optimal.
#
# Here we use the median score of the test-independent
# distribution only as a simple demonstration.
#
# For your final experiment, use a validation set.

threshold = np.median(
    y_score
)

y_pred = (
    y_score >= threshold
).astype(int)


# ============================================================
# 18. METRICS
# ============================================================

accuracy = accuracy_score(
    y_true,
    y_pred
)

precision = precision_score(
    y_true,
    y_pred,
    zero_division=0
)

recall = recall_score(
    y_true,
    y_pred,
    zero_division=0
)

f1 = f1_score(
    y_true,
    y_pred,
    zero_division=0
)

auroc = roc_auc_score(
    y_true,
    y_score
)


# ============================================================
# 19. CONFUSION MATRIX
# ============================================================

cm = confusion_matrix(
    y_true,
    y_pred
)

tn, fp, fn, tp = cm.ravel()

fpr = (
    fp / (fp + tn)
    if (fp + tn) > 0
    else 0.0
)


# ============================================================
# 20. PRINT RESULTS
# ============================================================

print(
    "\n============================================================"
)

print(
    "DETECTGPT RESULTS"
)

print(
    "============================================================"
)

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

print(
    f"FPR      : {fpr:.4f}"
)

print(
    "\nThreshold:",
    threshold
)

print(
    "\nConfusion Matrix:"
)

print(
    cm
)

print(
    "\nClassification Report:"
)

print(
    classification_report(
        y_true,
        y_pred,
        target_names=[
            "Human",
            "AI-generated"
        ],
        zero_division=0
    )
)


# ============================================================
# 21. SAVE RESULTS
# ============================================================

results.to_csv(
    "DetectGPT_HumanVsGPT4_results.csv",
    index=False,
    encoding="utf-8-sig"
)

print(
    "\nResults saved to:"
    " DetectGPT_HumanVsGPT4_results.csv"
)


# ============================================================
# 22. SAVE SUMMARY
# ============================================================

summary = pd.DataFrame(
    [
        {
            "Method": "DetectGPT",
            "Dataset": "HumanVsGPT-4",
            "Scoring Model": SCORING_MODEL_NAME,
            "Perturbation Model": MASK_FILLING_MODEL_NAME,
            "Perturbations": N_PERTURBATIONS,
            "Accuracy": accuracy,
            "Precision": precision,
            "Recall": recall,
            "F1": f1,
            "AUROC": auroc,
            "FPR": fpr
        }
    ]
)

summary.to_csv(
    "DetectGPT_HumanVsGPT4_summary.csv",
    index=False,
    encoding="utf-8-sig"
)

print(
    "\nSummary saved."
)