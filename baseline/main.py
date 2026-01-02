import timeit
import warnings
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.naive_bayes import MultinomialNB

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, confusion_matrix, ConfusionMatrixDisplay  # NEW IMPORTS
from sklearn.exceptions import ConvergenceWarning

warnings.filterwarnings("ignore", category=ConvergenceWarning, module="sklearn")

t0 = timeit.default_timer()


def load_and_transform_data():
    try:
        df_raw = pd.read_json('./dataset/synthetic_data_for_classification.jsonl', lines=True)
    except ValueError:
        print("Warning: Could not load data. Ensure file path is correct.")
        return [], []

    df_raw.fillna("", inplace=True)

    df_raw['anchor_text'] = df_raw['anchor_text'].astype(str)
    df_raw['text_a'] = df_raw['text_a'].astype(str)
    df_raw['text_b'] = df_raw['text_b'].astype(str)

    df1 = pd.DataFrame({
        'pair_text': df_raw['anchor_text'] + " [SEP] " + df_raw['text_a'],
        'label': df_raw['text_a_is_closer'].astype(int)
    })

    df2 = pd.DataFrame({
        'pair_text': df_raw['anchor_text'] + " [SEP] " + df_raw['text_b'],
        'label': (~df_raw['text_a_is_closer']).astype(int)
    })

    df_combined = pd.concat([df1, df2]).reset_index(drop=True)
    raw_documents = df_combined['pair_text'].values
    y_target = df_combined['label'].values

    return raw_documents, y_target


raw_X, y = load_and_transform_data()

print("Vectorizing text data...")
vectorizer = TfidfVectorizer(max_features=10000, stop_words='english')
X = vectorizer.fit_transform(raw_X)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, random_state=42, stratify=y, test_size=0.1
)

train_samples, n_features = X_train.shape
n_classes = 2

print(
    "Dataset Narrative_Sim, train_samples=%i, n_features=%i, n_classes=%i"
    % (train_samples, n_features, n_classes)
)

solver = "saga"

models = {
    "binary": {"name": "Binary Logistic", "iters": [10000]},
    "bayes": {"name": "Multinomial bayes", "iters": [10000]}
}


for model in models:
    accuracies = []
    times = [0]

    precisions = [0]
    recalls = [0]

    model_params = models[model]

    for this_max_iter in model_params["iters"]:
        print(
            "[model=%s, solver=%s] Number of epochs: %s"
            % (model_params["name"], solver, this_max_iter)
        )

        if model == "bayes":
            clf = MultinomialNB(alpha=.01)
        else:
            clf = LogisticRegression(
                penalty='elasticnet',
                l1_ratio=1,
                solver=solver,
                max_iter=this_max_iter,
                random_state=42,
            )

        t1 = timeit.default_timer()
        clf.fit(X_train, y_train)
        train_time = timeit.default_timer() - t1

        y_pred = clf.predict(X_test)

        accuracy = np.sum(y_pred == y_test) / y_test.shape[0]
        precision = precision_score(y_test, y_pred, zero_division=0)
        recall = recall_score(y_test, y_pred, zero_division=0)
        cm = confusion_matrix(y_test, y_pred)

        accuracies.append(accuracy)
        times.append(train_time)
        precisions.append(precision)
        recalls.append(recall)

    models[model]["times"] = times
    models[model]["accuracies"] = accuracies
    models[model]["precisions"] = precisions
    models[model]["recalls"] = recalls
    models[model]["confusion_matrix"] = cm

    print(f"--- Results for {model} ---")
    print("Test Accuracy:  %.4f" % accuracies[-1])
    print("Test Precision: %.4f" % precisions[-1])
    print("Test Recall:    %.4f" % recalls[-1])

    print(
        "Run time (%i epochs): %.2f s"
        % (model_params["iters"][-1], times[-1])
    )

    print("\nConfusion Matrix:")
    print(cm)
    print("-" * 30)

fig1 = plt.figure(figsize=(10, 5))

#
# for model in models:
#     name = models[model]["name"]
#     times = models[model]["times"]
#     accuracies = models[model]["accuracies"]
#     ax1.plot(times, accuracies, marker="o", label="Model: %s" % name)
#     ax1.set_xlabel("Train time (s)")
#     ax1.set_ylabel("Test accuracy")
#
# ax1.legend()
# ax1.set_title("Training Speed vs Accuracy")

# Plot 2: Confusion Matrix (New)
for index,model in enumerate(models):
    ax1 = fig1.add_subplot(121+index)
    cm= models[model]["confusion_matrix"]

    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Dissimilar (0)", "Similar (1)"])
    disp.plot(ax=ax1, cmap='Blues', values_format='d')
    ax1.set_title(f"Confusion Matrix: {models[model]['name']}")

    fig1.tight_layout()
    fig1.subplots_adjust(top=0.85)

run_time = timeit.default_timer() - t0
print("Total pipeline run in %.3f s" % run_time)
plt.show()
