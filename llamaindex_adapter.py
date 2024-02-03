import json

from llama_index import SimpleDirectoryReader
from llama_index.node_parser import SentenceSplitter
from llama_index.embeddings import resolve_embed_model, LinearAdapterEmbeddingModel
from llama_index.schema import MetadataMode
from llama_index.finetuning import (
    EmbeddingQAFinetuneDataset,
    EmbeddingAdapterFinetuneEngine,
)
import torch
from sentence_transformers import SentenceTransformer

from utils.embeddings_utils import cosine_similarity  # for embeddings
from eval_utils import evaluate, display_results
from CustomEmbedding import CustomSenteceTransformerEmbedding

TRAIN_CORPUS_FPATH = "./data/10_train_dataset.json"
VAL_CORPUS_FPATH = "./data/10_val_dataset.json"
base_embed_model = CustomSenteceTransformerEmbedding('DMetaSoul/Dmeta-embedding')
train_dataset = EmbeddingQAFinetuneDataset.from_json(TRAIN_CORPUS_FPATH)
val_dataset = EmbeddingQAFinetuneDataset.from_json(VAL_CORPUS_FPATH)

def load_corpus(files, verbose=False):
    if verbose:
        print(f"Loading files {files}")

    reader = SimpleDirectoryReader(input_files=files)
    docs = reader.load_data()
    if verbose:
        print(f"Loaded {len(docs)} docs")

    parser = SentenceSplitter()
    nodes = parser.get_nodes_from_documents(docs, show_progress=verbose)

    if verbose:
        print(f"Parsed {len(nodes)} nodes")

    return nodes

def finetune():
    finetune_engine = EmbeddingAdapterFinetuneEngine(
        train_dataset,
        base_embed_model,
        model_output_path="dmeta-finetuned-20",
        # bias=True,
        epochs=20,
        verbose=True,
        # optimizer_class=torch.optim.SGD,
        # optimizer_params={"lr": 0.01}
    )
    finetune_engine.finetune()
    embed_model = finetune_engine.get_finetuned_model()
    return embed_model

def main():
    finetuned_model = finetune()
    ft_embed_model = LinearAdapterEmbeddingModel(base_embed_model, 'dmeta-finetuned')
    dmeta = CustomSenteceTransformerEmbedding("DMetaSoul/Dmeta-embedding")
    dmeta_val_results = evaluate(val_dataset, dmeta)
    ft_val_results = evaluate(val_dataset, ft_embed_model)
    display_results(
        ["dmeat", "ft-dmeta"], [dmeta_val_results, ft_val_results]
    )

main()
