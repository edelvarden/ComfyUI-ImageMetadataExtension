import os
import folder_paths
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


def extract_embedding_names(text, input_data):
    embedding_names, _ = _extract_embedding_names(text, input_data)
    return [os.path.basename(embedding_name) for embedding_name in embedding_names]

def extract_embedding_hashes(text, input_data):
    embedding_names, clip = _extract_embedding_names(text, input_data)
    embedding_hashes = [calc_hash(get_embedding_file_path(name, clip)) for name in embedding_names]
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
    clip_ = input_data[0]["clip"][0]
    clip = get_clip_from_tokenizer(clip_.tokenizer) if clip_ else None
    embedding_identifier = clip.embedding_identifier if clip and hasattr(clip, "embedding_identifier") else "embedding:"

    # Ensure text is a string
    if not isinstance(text, str):
        text = "".join(str(item) if item else "" for item in text)
    
    text = escape_important(text)
    parsed_weights = token_weights(text, 1.0)

    # Tokenize and extract embedding names
    embedding_names = [
        word[len(embedding_identifier):].strip("\n")
        for weighted_segment, _ in parsed_weights
        for word in unescape_important(weighted_segment).replace("\n", " ").split(" ")
        if word.startswith(embedding_identifier) and clip and clip.embedding_directory
    ]

    return embedding_names, clip
