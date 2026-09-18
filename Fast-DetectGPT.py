# ============================================================
# Fast-DetectGPT Baseline
# Efficient Zero-Shot Detection of Machine-Generated Text
#
# Dataset:
#     HumanVsGPT-4.json
#
# Labels:
#     Human = 0
#     GPT-4 = 1
#
# Method:
#     Fast-DetectGPT conditional probability curvature
#
# Sampling Model:
#     gpt2-xl
#
# Scoring Model:
#     gpt2-xl
#
# ============================================================


# ============================================================
# 1. IMPORT LIBRARIES
# ============================================================

import os
import json
import random
import warnings

import numpy as np
import pandas as pd
import torch

from tqdm import tqdm

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM
)

from sklearn.model_selection import train_test_split

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report
)

warnings.filterwarnings("ignore")


# ============================================================
# 2. CONFIGURATION
# ============================================================

DATASET_PATH = (
    "HumanVsGPT-4.json"
)

# ------------------------------------------------------------
# Sampling model
# ------------------------------------------------------------

SAMPLING_MODEL_NAME = (
    "gpt2-xl"
)

# ------------------------------------------------------------
# Scoring model
# ------------------------------------------------------------

SCORING_MODEL_NAME = (
    "gpt2-xl"
)

# ------------------------------------------------------------
# Dataset split
# ------------------------------------------------------------

TEST_SIZE = 0.20

RANDOM_SEED = 42

# ------------------------------------------------------------
# Device
# ------------------------------------------------------------

DEVICE = (
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

print(
    "Device:",
    DEVICE
)


# ============================================================
# 3. RANDOM SEED
# ============================================================

random.seed(
    RANDOM_SEED
)

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
# 7. LOAD SCORING TOKENIZER
# ============================================================

print(
    "\nLoading scoring tokenizer..."
)

scoring_tokenizer = (
    AutoTokenizer.from_pretrained(
        SCORING_MODEL_NAME
    )
)

if scoring_tokenizer.pad_token is None:

    scoring_tokenizer.pad_token = (
        scoring_tokenizer.eos_token
    )


# ============================================================
# 8. LOAD SCORING MODEL
# ============================================================

print(
    "Loading scoring model..."
)

scoring_model = (
    AutoModelForCausalLM.from_pretrained(
        SCORING_MODEL_NAME
    )
)

scoring_model.to(
    DEVICE
)

scoring_model.eval()


# ============================================================
# 9. LOAD SAMPLING MODEL
# ============================================================

if (
    SAMPLING_MODEL_NAME
    == SCORING_MODEL_NAME
):

    print(
        "\nSampling model and scoring model are identical."
    )

    sampling_tokenizer = (
        scoring_tokenizer
    )

    sampling_model = (
        scoring_model
    )

else:

    print(
        "\nLoading sampling tokenizer..."
    )

    sampling_tokenizer = (
        AutoTokenizer.from_pretrained(
            SAMPLING_MODEL_NAME
        )
    )

    if sampling_tokenizer.pad_token is None:

        sampling_tokenizer.pad_token = (
            sampling_tokenizer.eos_token
        )

    print(
        "Loading sampling model..."
    )

    sampling_model = (
        AutoModelForCausalLM.from_pretrained(
            SAMPLING_MODEL_NAME
        )
    )

    sampling_model.to(
        DEVICE
    )

    sampling_model.eval()


# ============================================================
# 10. GET CONDITIONAL PROBABILITY CURVATURE
# ============================================================

def get_sampling_discrepancy(
    logits_ref,
    logits_score,
    labels
):

    # --------------------------------------------------------
    # Convert logits to probabilities
    # --------------------------------------------------------

    ref_probs = torch.softmax(
        logits_ref,
        dim=-1
    )

    # --------------------------------------------------------
    # Log probabilities
    # --------------------------------------------------------

    log_probs_score = (
        torch.log_softmax(
            logits_score,
            dim=-1
        )
    )

    # --------------------------------------------------------
    # Reference probability for the actual token
    # --------------------------------------------------------

    ref_probs_actual = (
        ref_probs.gather(
            2,
            labels.unsqueeze(-1)
        ).squeeze(-1)
    )

    # --------------------------------------------------------
    # Scoring log probability
    # --------------------------------------------------------

    log_probs_actual = (
        log_probs_score.gather(
            2,
            labels.unsqueeze(-1)
        ).squeeze(-1)
    )

    # --------------------------------------------------------
    # Expected scoring log probability
    # under the sampling distribution
    # --------------------------------------------------------

    expected_log_prob = (
        (
            ref_probs
            * log_probs_score
        )
        .sum(
            dim=-1
        )
    )

    # --------------------------------------------------------
    # Conditional probability discrepancy
    # --------------------------------------------------------

    discrepancy = (
        log_probs_actual
        - expected_log_prob
    )

    # --------------------------------------------------------
    # Normalize
    # --------------------------------------------------------

    mean = discrepancy.mean()

    std = discrepancy.std()

    if std.item() < 1e-8:

        return mean

    return (
        mean / std
    )


# ============================================================
# 11. COMPUTE FAST-DETECTGPT SCORE
# ============================================================

def fast_detectgpt_score(
    text
):

    if not text.strip():

        return 0.0

    # --------------------------------------------------------
    # Score text using scoring tokenizer
    # --------------------------------------------------------

    score_tokens = (
        scoring_tokenizer(
            text,
            truncation=True,
            max_length=1024,
            return_tensors="pt",
            padding=True,
            return_token_type_ids=False
        )
    )

    score_tokens = {
        key: value.to(DEVICE)
        for key, value in score_tokens.items()
    }

    labels = (
        score_tokens[
            "input_ids"
        ][:, 1:]
    )

    # --------------------------------------------------------
    # Need at least two tokens
    # --------------------------------------------------------

    if labels.shape[1] == 0:

        return 0.0

    # --------------------------------------------------------
    # Scoring model logits
    # --------------------------------------------------------

    with torch.no_grad():

        scoring_outputs = (
            scoring_model(
                **score_tokens
            )
        )

        logits_score = (
            scoring_outputs.logits[
                :,
                :-1,
                :
            ]
        )

    # --------------------------------------------------------
    # Sampling model logits
    # --------------------------------------------------------

    if (
        SAMPLING_MODEL_NAME
        == SCORING_MODEL_NAME
    ):

        logits_ref = (
            logits_score
        )

    else:

        sample_tokens = (
            sampling_tokenizer(
                text,
                truncation=True,
                max_length=1024,
                return_tensors="pt",
                padding=True,
                return_token_type_ids=False
            )
        )

        sample_tokens = {
            key: value.to(DEVICE)
            for key, value
            in sample_tokens.items()
        }

        # ----------------------------------------------------
        # Verify tokenization compatibility
        # ----------------------------------------------------

        sample_labels = (
            sample_tokens[
                "input_ids"
            ][:, 1:]
        )

        if not torch.equal(
            sample_labels,
            labels
        ):

            raise ValueError(
                "Sampling and scoring tokenizers "
                "produce different token IDs. "
                "Use compatible tokenizers/models."
            )

        with torch.no_grad():

            sampling_outputs = (
                sampling_model(
                    **sample_tokens
                )
            )

            logits_ref = (
                sampling_outputs.logits[
                    :,
                    :-1,
                    :
                ]
            )

    # --------------------------------------------------------
    # Compute criterion
    # --------------------------------------------------------

    criterion = (
        get_sampling_discrepancy(
            logits_ref,
            logits_score,
            labels
        )
    )

    return float(
        criterion.item()
    )


# ============================================================
# 12. RUN FAST-DETECTGPT
# ============================================================

scores = []
labels = []

print(
    "\n============================================================"
)

print(
    "Running Fast-DetectGPT"
)

print(
    "============================================================"
)

for index, row in tqdm(
    df_test.iterrows(),
    total=len(df_test)
):

    text = row[
        "text"
    ]

    label = row[
        "label"
    ]

    try:

        score = (
            fast_detectgpt_score(
                text
            )
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
# 13. RESULTS DATAFRAME
# ============================================================

results = pd.DataFrame(
    {
        "text":
            df_test[
                "text"
            ].values,

        "label":
            labels,

        "FastDetectGPT_Score":
            scores
    }
)


# ============================================================
# 14. SCORE
# ============================================================

y_true = results[
    "label"
].values

y_score = results[
    "FastDetectGPT_Score"
].values


# ============================================================
# 15. DETERMINE THRESHOLD
# ============================================================

# For a simple experiment we use the median.
#
# IMPORTANT:
# Do not use this as the final methodology if you are
# reporting a rigorous baseline comparison.
#
# Instead:
#   1. train/validation/test split
#   2. select threshold on validation
#   3. evaluate once on test

threshold = np.median(
    y_score
)

y_pred = (
    y_score >= threshold
).astype(int)


# ============================================================
# 16. METRICS
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
# 17. CONFUSION MATRIX
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
# 18. PRINT RESULTS
# ============================================================

print(
    "\n============================================================"
)

print(
    "FAST-DETECTGPT RESULTS"
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
# 19. SAVE RESULTS
# ============================================================

results.to_csv(
    "FastDetectGPT_HumanVsGPT4_results.csv",
    index=False,
    encoding="utf-8-sig"
)

print(
    "\nResults saved to:"
    " FastDetectGPT_HumanVsGPT4_results.csv"
)


# ============================================================
# 20. SAVE SUMMARY
# ============================================================

summary = pd.DataFrame(
    [
        {
            "Method":
                "Fast-DetectGPT",

            "Dataset":
                "HumanVsGPT-4",

            "Sampling Model":
                SAMPLING_MODEL_NAME,

            "Scoring Model":
                SCORING_MODEL_NAME,

            "Accuracy":
                accuracy,

            "Precision":
                precision,

            "Recall":
                recall,

            "F1":
                f1,

            "AUROC":
                auroc,

            "FPR":
                fpr
        }
    ]
)

summary.to_csv(
    "FastDetectGPT_HumanVsGPT4_summary.csv",
    index=False,
    encoding="utf-8-sig"
)

print(
    "\nSummary saved."
)