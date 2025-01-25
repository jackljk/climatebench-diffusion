import hashlib
import json
import os
from datetime import datetime
from typing import Dict, Any, Union

import numpy as np
import h5py
from pathlib import Path
from tqdm import tqdm
import torch
from transformers import BertModel, BertTokenizer, AutoModelForCausalLM, AutoTokenizer
from src.utilities.utils import get_logger

log = get_logger(__name__)

def get_llama_embedding(text, tokenizer=None, model=None, pooling=True, last_layer=True):
    """
    get the embedding of the text using the llama model and tokenizer
    text: str, the text to be embedded
    tokenizer: llama tokenizer
    model: llama model
    pooling: bool, whether to use mean pooling or take the last token's embedding
    last_layer: bool, whether to use the last layer or the middle layer
    """
    inputs = tokenizer(text, return_tensors="pt")

    with torch.inference_mode():
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        outputs = model(**inputs, output_hidden_states=True)
        hidden_states = outputs.hidden_states

    if last_layer:
        target_hidden_state = hidden_states[-1]

    else:
        # take the middle one
        target_hidden_state = hidden_states[len(hidden_states) // 2]

    # Mask padding tokens
    attention_mask = inputs['attention_mask']
    masked_embeddings = target_hidden_state * attention_mask.unsqueeze(-1)

    if pooling:
        # Compute mean pooling (ignoring padding)
        sentence_embedding = masked_embeddings.sum(dim=1) / attention_mask.sum(dim=1, keepdim=True)

    else:
        # take the last token's embedding
        sentence_embedding = masked_embeddings[:, -1, :]

    return sentence_embedding[0].cpu().numpy()

def get_bert_embeddings(text, tokenizer, model, max_length=512):
    # Tokenize input text
    tokens = tokenizer.encode(text, add_special_tokens=True, max_length=max_length, truncation=True)
    # Convert tokens to tensor
    tokens_tensor = torch.tensor([tokens]).to(model.device)
    # Get BERT model output
    with torch.inference_mode():
        outputs = model(tokens_tensor)
        # Extract embeddings for [CLS] token (first token)
        embeddings = outputs[0][:, 0, :].squeeze().cpu().numpy()
    return embeddings


def get_dict_hash(d, length=8):
    """Create deterministic hash from a dictionary."""
    dhash = hashlib.md5(json.dumps(d, sort_keys=True).encode()).hexdigest()
    return dhash[:length]

def get_or_create_embeddings(corpus, model_name, save_dir, force_recreate=False, metadata: Dict = None):
    """
    Load embeddings if they exist, otherwise create and save them.

    Args:
        corpus: List of texts to embed
        model_name: Name of the model to use for embeddings (e.g. 'bert-base-uncased', "Meta-Llama-3.1-8B")
        save_dir: Directory to save embeddings
        force_recreate: Whether to recreate embeddings even if they exist
        metadata: Optional metadata to save with the embeddings (and verify when loading)

    Returns:
        List of numpy arrays containing embeddings
    """
    save_path = None
    if save_dir is not None:
        save_dir = Path(save_dir)
        save_filename = f"{model_name}"
        if metadata is not None:
            save_filename += str(get_dict_hash(metadata))
        save_filename += "-embeddings.h5"
        save_path = save_dir / save_filename
        save_path.parent.mkdir(parents=True, exist_ok=True)
        print(f"save_path: {save_path}")

    # Try to load existing embeddings
    if save_dir is not None and save_path.exists() and not force_recreate:
        log.info(f"Loading existing embeddings from {save_path}")
        try:
            with h5py.File(save_path, 'r') as f:
                if metadata is not None:
                    for key, value in metadata.items():
                        if f.attrs.get(key) != value:
                            log.warning(f"Metadata mismatch: {key}={f.attrs.get(key)} vs {value}")
                            raise ValueError("Metadata mismatch")
                num_saved = f.attrs.get('num_embeddings', 0)
                if num_saved == len(corpus):  # Only use if complete
                    return [f['embeddings'][i] for i in range(num_saved)]
                else:
                    log.warning(f"Found incomplete embeddings ({num_saved} vs {len(corpus)} needed)")
        except Exception as e:
            log.warning(f"Error loading embeddings: {e}")

    # Create new embeddings
    log.info(f"Computing {model_name} embeddings for text data...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    load_kwargs = dict()
    if "bert" in model_name.lower():
        if os.environ.get("PSCRATCH") is not None:
            maybe_from = os.path.join(os.environ["PSCRATCH"], "huggingface", model_name)
            if os.path.exists(maybe_from):
                # Useful on SLURM when the model is already downloaded and network is not available/slow
                model_name = maybe_from
                log.info(f"Loading BERT model locally from {model_name}")

        model_class = BertModel
        tokenizer_class = BertTokenizer
        embedding_func = get_bert_embeddings
    elif "llama" in model_name.lower():
        from huggingface_hub import login
        from huggingface_hub.hf_api import HfFolder

        token = os.environ.get("HUGGINGFACE_TOKEN") or os.environ.get("HF_TOKEN")
        if token is None:
            raise ValueError("Please set the HUGGINGFACE_TOKEN environment variable to use LLama.")
        if os.environ.get("HF_HOME") is None:
            os.environ["HF_HOME"] = "~/.cache/huggingface"
        # logout()
        login(token=token)
        HfFolder.save_token(token)

        cache_dir = "/trunk/model-hub"
        if os.environ.get("PSCRATCH") is not None:
            cache_dir = os.path.join(os.environ["PSCRATCH"], "huggingface", model_name)
        model_class = AutoModelForCausalLM
        tokenizer_class = AutoTokenizer
        load_kwargs["cache_dir"] = cache_dir
        embedding_func = get_llama_embedding
    else:
        raise ValueError(f"Unknown model name {model_name}")

    model = model_class.from_pretrained(model_name, **load_kwargs)
    model.eval()
    model.to(device)
    tokenizer = tokenizer_class.from_pretrained(model_name, **load_kwargs)

    text_features = []
    # embeddings = corpus.apply(lambda x: get_bert_embeddings(x, tokenizer=tokenizer, model=model, device=device))
    for x in tqdm(corpus, desc=f"{model_name} embeddings", total=len(corpus)):
        embedding = embedding_func(x, tokenizer=tokenizer, model=model, pooling=True, last_layer=True)
        embedding = np.array(embedding, dtype=np.float32)
        text_features.append(embedding)

    embedding_dim = len(text_features[0])
    try:
        # Save embeddings
        with h5py.File(save_path, 'w') as f:
            embeddings_array = np.stack(text_features, axis=0)
            f.create_dataset(
                'embeddings',
                data=embeddings_array,
                compression='gzip',
                compression_opts=4,
                chunks=(min(1000, len(embeddings_array)), embedding_dim)
            )
            f.attrs['num_embeddings'] = len(embeddings_array)
            f.attrs['embedding_dim'] = embedding_dim
            if metadata is not None:
                for key, value in metadata.items():
                    f.attrs[key] = convert_to_saveable_type(value)

        log.info(f"Saved new embeddings to {save_path}")

    except Exception as e:
        log.error(f"Error saving embeddings: {e}")
        raise
    # del model
    return text_features

def convert_to_saveable_type(value: Any) -> Union[str, int, float, np.ndarray]:
    """Convert Python objects to HDF5-compatible types."""
    if isinstance(value, (str, int, float, np.ndarray)):
        return value
    elif isinstance(value, (datetime, np.datetime64)):
        return str(value)
    elif isinstance(value, (list, tuple)):
        return np.array(value)
    else:
        return str(value)