import os
from comfy.sd1_clip import expand_directory_list

def get_embedding_file_path(embedding_name, clip):
    """
    Resolves the file path for an embedding by searching directories and checking file extensions.

    Args:
        embedding_name (str): The name of the embedding file (without an extension).
        clip (object): Object containing `embedding_directory` specifying directories to search.

    Returns:
        str or None: Full path to the embedding file if found, otherwise None.
    """
    # Validate embedding_directory
    embedding_directory = getattr(clip, "embedding_directory", None)
    if not embedding_directory:
        return None
    
    embedding_directory = (
        [embedding_directory] if isinstance(embedding_directory, str) else embedding_directory
    )

    # Expand directories
    try:
        embedding_directory = expand_directory_list(embedding_directory)
    except Exception:
        return None

    if not embedding_directory:
        return None

    extensions = ["", ".safetensors", ".pt", ".bin"]

    for embed_dir in embedding_directory:
        embed_dir = os.path.abspath(embed_dir)
        if not os.path.isdir(embed_dir):
            continue  # Skip invalid directories

        for ext in extensions:
            candidate_path = os.path.join(embed_dir, embedding_name + ext)
            if os.path.isfile(candidate_path):
                return candidate_path  # Return immediately when a valid file is found

    return None  # Return None if no file is found
