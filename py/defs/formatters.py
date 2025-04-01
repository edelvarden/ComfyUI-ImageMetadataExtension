import os
import folder_paths
import re
from ..utils.hash import calc_hash
from ..utils.embedding import get_embedding_file_path
from comfy.sd1_clip import escape_important, token_weights, unescape_important
from comfy.sd1_clip import SD1Tokenizer
from comfy.text_encoders.sd2_clip import SD2Tokenizer
from comfy.text_encoders.sd3_clip import SD3Tokenizer
from comfy.text_encoders.flux import FluxTokenizer
from comfy.sdxl_clip import SDXLTokenizer

cache_model_hash = {}

# Generalized hash calculation for different folder types
def calc_hash_for_type(folder_type, model_name):
    try:
        filename = folder_paths.get_full_path(folder_type, model_name)
        return calc_hash(filename)
    except Exception as e:
        return ""  # Return empty string if unable to calculate hash

# Replacing calc_model_hash, calc_vae_hash, calc_lora_hash, and calc_unet_hash
def calc_model_hash(model_name, input_data):
    return calc_hash_for_type("checkpoints", model_name)

def calc_vae_hash(model_name, input_data):
    return calc_hash_for_type("vae", model_name)

def calc_lora_hash(model_name, input_data):
    return calc_hash_for_type("loras", model_name)

def calc_unet_hash(model_name, input_data):
    return calc_hash_for_type("unet", model_name)


def convert_skip_clip(stop_at_clip_layer, input_data):
    return stop_at_clip_layer * -1


SCALING_FACTOR = 8

def get_scaled_width(scaled_by, input_data):
    samples = input_data[0]["samples"][0]["samples"]
    return round(samples.shape[3] * scaled_by * SCALING_FACTOR)

def get_scaled_height(scaled_by, input_data):
    samples = input_data[0]["samples"][0]["samples"]
    return round(samples.shape[2] * scaled_by * SCALING_FACTOR)


embedding_pattern = re.compile(r"embedding:\(?([^\s),]+)\)?")

def _extract_embedding_names_from_text(text):
    embedding_identifier = "embedding:"

    # Check if 'embedding:' exists in the text
    if embedding_identifier not in text:
        return []
    
    # Extract matches
    embedding_names = [match.group(1) for match in embedding_pattern.finditer(text)]

    return embedding_names

def extract_embedding_names(text, clip):
    return _extract_embedding_names_from_text(text)

def extract_embedding_hashes(text, input_data):
    embedding_names, clip = _extract_embedding_names(text, input_data)

    if not clip:
        return []

    embedding_hashes = [
        calc_hash(get_embedding_file_path(name, clip)) if get_embedding_file_path(name, clip) else ""
        for name in embedding_names
    ]
    return embedding_hashes

# Helper function to get clip from the tokenizer
def get_clip_from_tokenizer(tokenizer):
    if isinstance(tokenizer, (SD1Tokenizer, SD3Tokenizer, SDXLTokenizer, FluxTokenizer)):
        return tokenizer.clip_l
    elif isinstance(tokenizer, SD2Tokenizer):
        return tokenizer.clip_h
    return None

# Extract embedding names from text
def _extract_embedding_names(text, input_data):
    clip_ = input_data[0].get("clip", [None])[0]
    clip = get_clip_from_tokenizer(clip_.tokenizer) if clip_ else None

    if not clip or not hasattr(clip, "embedding_directory"):
        return [], None

    # Extract embedding names
    embedding_names = _extract_embedding_names_from_text(text)

    # Escape and tokenize if necessary
    text = escape_important(text)
    parsed_weights = token_weights(text, 1.0)

    # Add matches from tokenized and escaped text
    embedding_names += [
        match.group(1) for segment, _ in parsed_weights
        for match in re.finditer(embedding_pattern, unescape_important(segment))
    ]

    return embedding_names, clip
