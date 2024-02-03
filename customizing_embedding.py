# imports
from typing import List, Tuple  # for type hints

import numpy as np  # for manipulating arrays
import pandas as pd  # for manipulating data in dataframes
import pickle  # for saving the embeddings cache
import plotly.express as px  # for plots
import random  # for generating run IDs
from sklearn.model_selection import train_test_split  # for splitting train & test data
import torch  # for matrix optimization
from sentence_transformers import SentenceTransformer

from utils.embeddings_utils import cosine_similarity  # for embeddings

# Set display option to show all columns
pd.set_option('display.max_columns', None)

# input parameters
embedding_cache_path = "data/snli_embedding_cache.pkl"  # embeddings will be saved/loaded here
default_embedding_engine = "all-MiniLM-L6-v2"
num_pairs_to_embed = 1000  # 1000 is arbitrary
local_dataset_path = "data/snli_1.0_train_2k.csv"  # download from: https://nlp.stanford.edu/projects/snli/
embedding_model = SentenceTransformer(default_embedding_engine)

def get_embedding(text: str, **kwargs) -> List[List[float]]:
    # replace newlines, which can negatively affect performance.
    text = text.replace("\n", " ")
    embeddings = embedding_model.encode(text)
    return embeddings

def process_input_data(df: pd.DataFrame) -> pd.DataFrame:
    # you can customize this to preprocess your own dataset
    # output should be a dataframe with 3 columns: text_1, text_2, label (1 for similar, -1 for dissimilar)
    df["label"] = df["gold_label"]
    df = df[df["label"].isin(["entailment"])]
    df["label"] = df["label"].apply(lambda x: {"entailment": 1, "contradiction": -1}[x])
    df = df.rename(columns={"sentence1": "text_1", "sentence2": "text_2"})
    df = df[["text_1", "text_2", "label"]]
    df = df.head(num_pairs_to_embed)
    return df

# generate negatives
def dataframe_of_negatives(dataframe_of_positives: pd.DataFrame) -> pd.DataFrame:
    """Return dataframe of negative pairs made by combining elements of positive pairs."""
    texts = set(dataframe_of_positives["text_1"].values) | set(
        dataframe_of_positives["text_2"].values
    )
    all_pairs = {(t1, t2) for t1 in texts for t2 in texts if t1 < t2}
    positive_pairs = set(
        tuple(text_pair)
        for text_pair in dataframe_of_positives[["text_1", "text_2"]].values
    )
    negative_pairs = all_pairs - positive_pairs
    df_of_negatives = pd.DataFrame(list(negative_pairs), columns=["text_1", "text_2"])
    df_of_negatives["label"] = -1
    return df_of_negatives

# load data
df = pd.read_csv(local_dataset_path)

# process input data
df = process_input_data(df)  # this demonstrates training data containing only positives

# split data into train and test sets
test_fraction = 0.5  # 0.5 is fairly arbitrary
random_seed = 123  # random seed is arbitrary, but is helpful in reproducibility
train_df, test_df = train_test_split(
    df, test_size=test_fraction, stratify=df["label"], random_state=random_seed
)
train_df.loc[:, "dataset"] = "train"
test_df.loc[:, "dataset"] = "test"

negatives_per_positive = (
    1  # it will work at higher values too, but more data will be slower
)
# generate negatives for training dataset
train_df_negatives = dataframe_of_negatives(train_df)
train_df_negatives["dataset"] = "train"
# generate negatives for test dataset
test_df_negatives = dataframe_of_negatives(test_df)
test_df_negatives["dataset"] = "test"
# sample negatives and combine with positives
train_df = pd.concat(
    [
        train_df,
        train_df_negatives.sample(
            n=len(train_df) * negatives_per_positive, random_state=random_seed
        ),
    ]
)
test_df = pd.concat(
    [
        test_df,
        test_df_negatives.sample(
            n=len(test_df) * negatives_per_positive, random_state=random_seed
        ),
    ]
)

df = pd.concat([train_df, test_df])

embedding_cache = {}


# this function will get embeddings from the cache and save them there afterward
def get_embedding_with_cache(
    text: str,
    engine: str = default_embedding_engine,
    embedding_cache: dict = embedding_cache,
    embedding_cache_path: str = embedding_cache_path,
) -> list:
    if (text, engine) not in embedding_cache.keys():
        # if not in cache, call API to get embedding
        print(text, get_embedding(text))
        embedding_cache[(text, engine)] = get_embedding(text)
        # save embeddings cache to disk after each update
        with open(embedding_cache_path, "wb") as embedding_cache_file:
            pickle.dump(embedding_cache, embedding_cache_file)
    return embedding_cache[(text, engine)]

# create column of embeddings
for column in ["text_1", "text_2"]:
    df[f"{column}_embedding"] = df[column].apply(get_embedding_with_cache)

# create column of cosine similarity between embeddings
df["cosine_similarity"] = df.apply(
    lambda row: cosine_similarity(row["text_1_embedding"], row["text_2_embedding"]),
    axis=1,
)

# calculate accuracy (and its standard error) of predicting label=1 if similarity>x
# x is optimized by sweeping from -1 to 1 in steps of 0.01
def accuracy_and_se(cosine_similarity: float, labeled_similarity: int) -> Tuple[float]:
    accuracies = []
    for threshold_thousandths in range(-1000, 1000, 1):
        threshold = threshold_thousandths / 1000
        total = 0
        correct = 0
        for cs, ls in zip(cosine_similarity, labeled_similarity):
            total += 1
            if cs > threshold:
                prediction = 1
            else:
                prediction = -1
            if prediction == ls:
                correct += 1
        accuracy = correct / total
        accuracies.append(accuracy)
    a = max(accuracies)
    n = len(cosine_similarity)
    standard_error = (a * (1 - a) / n) ** 0.5  # standard error of binomial
    return a, standard_error


# check that training and test sets are balanced
px.histogram(
    df,
    x="cosine_similarity",
    color="label",
    barmode="overlay",
    width=500,
    facet_row="dataset",
).show()

for dataset in ["train", "test"]:
    data = df[df["dataset"] == dataset]
    a, se = accuracy_and_se(data["cosine_similarity"], data["label"])
    print(f"{dataset} accuracy: {a:0.1%} ± {1.96 * se:0.1%}")


def embedding_multiplied_by_matrix(
    embedding: List[float], matrix: torch.tensor
) -> np.array:
    embedding_tensor = torch.tensor(embedding).float()
    modified_embedding = embedding_tensor @ matrix
    modified_embedding = modified_embedding.detach().numpy()
    return modified_embedding


# compute custom embeddings and new cosine similarities
def apply_matrix_to_embeddings_dataframe(matrix: torch.tensor, df: pd.DataFrame):
    for column in ["text_1_embedding", "text_2_embedding"]:
        df[f"{column}_custom"] = df[column].apply(
            lambda x: embedding_multiplied_by_matrix(x, matrix)
        )
    df["cosine_similarity_custom"] = df.apply(
        lambda row: cosine_similarity(
            row["text_1_embedding_custom"], row["text_2_embedding_custom"]
        ),
        axis=1,
    )

def optimize_matrix(
    modified_embedding_length: int = 2048,  # in my brief experimentation, bigger was better (2048 is length of babbage encoding)
    batch_size: int = 100,
    max_epochs: int = 100,
    learning_rate: float = 100.0,  # seemed to work best when similar to batch size - feel free to try a range of values
    dropout_fraction: float = 0.0,  # in my testing, dropout helped by a couple percentage points (definitely not necessary)
    df: pd.DataFrame = df,
    print_progress: bool = True,
    save_results: bool = True,
) -> torch.tensor:
    """Return matrix optimized to minimize loss on training data."""
    run_id = random.randint(0, 2 ** 31 - 1)  # (range is arbitrary)
    # convert from dataframe to torch tensors
    # e is for embedding, s for similarity label
    def tensors_from_dataframe(
        df: pd.DataFrame,
        embedding_column_1: str,
        embedding_column_2: str,
        similarity_label_column: str,
    ) -> Tuple[torch.tensor]:
        e1 = np.stack(np.array(df[embedding_column_1].values))
        e2 = np.stack(np.array(df[embedding_column_2].values))
        s = np.stack(np.array(df[similarity_label_column].astype("float").values))

        e1 = torch.from_numpy(e1).float()
        e2 = torch.from_numpy(e2).float()
        s = torch.from_numpy(s).float()

        return e1, e2, s

    e1_train, e2_train, s_train = tensors_from_dataframe(
        df[df["dataset"] == "train"], "text_1_embedding", "text_2_embedding", "label"
    )
    e1_test, e2_test, s_test = tensors_from_dataframe(
        df[df["dataset"] == "test"], "text_1_embedding", "text_2_embedding", "label"
    )

    # create dataset and loader
    dataset = torch.utils.data.TensorDataset(e1_train, e2_train, s_train)
    train_loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=True
    )

    # define model (similarity of projected embeddings)
    def model(embedding_1, embedding_2, matrix, dropout_fraction=dropout_fraction):
        e1 = torch.nn.functional.dropout(embedding_1, p=dropout_fraction)
        e2 = torch.nn.functional.dropout(embedding_2, p=dropout_fraction)
        modified_embedding_1 = e1 @ matrix  # @ is matrix multiplication
        modified_embedding_2 = e2 @ matrix
        similarity = torch.nn.functional.cosine_similarity(
            modified_embedding_1, modified_embedding_2
        )
        return similarity

    # define loss function to minimize
    def mse_loss(predictions, targets):
        difference = predictions - targets
        return torch.sum(difference * difference) / difference.numel()

    # initialize projection matrix
    embedding_length = len(df["text_1_embedding"].values[0])
    matrix = torch.randn(
        embedding_length, modified_embedding_length, requires_grad=True
    )

    epochs, types, losses, accuracies, matrices = [], [], [], [], []
    for epoch in range(1, 1 + max_epochs):
        # iterate through training dataloader
        for a, b, actual_similarity in train_loader:
            # generate prediction
            predicted_similarity = model(a, b, matrix)
            # get loss and perform backpropagation
            loss = mse_loss(predicted_similarity, actual_similarity)
            loss.backward()
            # update the weights
            with torch.no_grad():
                matrix -= matrix.grad * learning_rate
                # set gradients to zero
                matrix.grad.zero_()
        # calculate test loss
        test_predictions = model(e1_test, e2_test, matrix)
        test_loss = mse_loss(test_predictions, s_test)

        # compute custom embeddings and new cosine similarities
        apply_matrix_to_embeddings_dataframe(matrix, df)

        # calculate test accuracy
        for dataset in ["train", "test"]:
            data = df[df["dataset"] == dataset]
            a, se = accuracy_and_se(data["cosine_similarity_custom"], data["label"])

            # record results of each epoch
            epochs.append(epoch)
            types.append(dataset)
            losses.append(loss.item() if dataset == "train" else test_loss.item())
            accuracies.append(a)
            matrices.append(matrix.detach().numpy())

            # optionally print accuracies
            if print_progress is True:
                print(
                    f"Epoch {epoch}/{max_epochs}: {dataset} accuracy: {a:0.1%} ± {1.96 * se:0.1%}"
                )

    data = pd.DataFrame(
        {"epoch": epochs, "type": types, "loss": losses, "accuracy": accuracies}
    )
    data["run_id"] = run_id
    data["modified_embedding_length"] = modified_embedding_length
    data["batch_size"] = batch_size
    data["max_epochs"] = max_epochs
    data["learning_rate"] = learning_rate
    data["dropout_fraction"] = dropout_fraction
    data[
        "matrix"
    ] = matrices  # saving every single matrix can get big; feel free to delete/change
    if save_results is True:
        data.to_csv(f"{run_id}_optimization_results.csv", index=False)

    return data


# example hyperparameter search
# I recommend starting with max_epochs=10 while initially exploring
results = []
max_epochs = 30
dropout_fraction = 0.2
for batch_size, learning_rate in [(10, 10), (100, 100), (1000, 1000)]:
    result = optimize_matrix(
        batch_size=batch_size,
        learning_rate=learning_rate,
        max_epochs=max_epochs,
        dropout_fraction=dropout_fraction,
        save_results=False,
    )
    results.append(result)



runs_df = pd.concat(results)

# plot training loss and test loss over time
px.line(
    runs_df,
    line_group="run_id",
    x="epoch",
    y="loss",
    color="type",
    hover_data=["batch_size", "learning_rate", "dropout_fraction"],
    facet_row="learning_rate",
    facet_col="batch_size",
    width=500,
).show()

# plot accuracy over time
px.line(
    runs_df,
    line_group="run_id",
    x="epoch",
    y="accuracy",
    color="type",
    hover_data=["batch_size", "learning_rate", "dropout_fraction"],
    facet_row="learning_rate",
    facet_col="batch_size",
    width=500,
).show()

# apply result of best run to original data
best_run = runs_df.sort_values(by="accuracy", ascending=False).iloc[0]
best_matrix = best_run["matrix"]
apply_matrix_to_embeddings_dataframe(best_matrix, df)

# plot similarity distribution BEFORE customization
px.histogram(
    df,
    x="cosine_similarity",
    color="label",
    barmode="overlay",
    width=500,
    facet_row="dataset",
).show()

test_df = df[df["dataset"] == "test"]
a, se = accuracy_and_se(test_df["cosine_similarity"], test_df["label"])
print(f"Test accuracy: {a:0.1%} ± {1.96 * se:0.1%}")

# plot similarity distribution AFTER customization
px.histogram(
    df,
    x="cosine_similarity_custom",
    color="label",
    barmode="overlay",
    width=500,
    facet_row="dataset",
).show()

a, se = accuracy_and_se(test_df["cosine_similarity_custom"], test_df["label"])
print(f"Test accuracy after customization: {a:0.1%} ± {1.96 * se:0.1%}")

def save_best_matrix(matrix):
    file_path = 'best_matrix.pkl'
    with open(file_path, 'wb') as file:
        pickle.dump(matrix, file)

save_best_matrix(best_matrix)
