"""
Hash utilities for the Agentic RAG system
"""

import hashlib
import os
from pathlib import Path
from typing import Union


def get_file_hash(file_path: Union[str, Path]) -> str:
    """
    Get SHA256 hash of a file

    Args:
        file_path: Path to the file

    Returns:
        SHA256 hash as hex string
    """
    if isinstance(file_path, str):
        file_path = Path(file_path)

    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    hash_sha256 = hashlib.sha256()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_sha256.update(chunk)

    return hash_sha256.hexdigest()


def get_text_hash(text: str) -> str:
    """
    Get SHA256 hash of text

    Args:
        text: Text to hash

    Returns:
        SHA256 hash as hex string
    """
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def get_directory_hash(directory: Union[str, Path]) -> str:
    """
    Get SHA256 hash of all files in a directory

    Args:
        directory: Path to the directory

    Returns:
        SHA256 hash as hex string
    """
    if isinstance(directory, str):
        directory = Path(directory)

    if not directory.exists():
        raise FileNotFoundError(f"Directory not found: {directory}")

    hash_sha256 = hashlib.sha256()

    # Walk through all files in directory
    for root, dirs, files in os.walk(directory):
        for file in files:
            file_path = Path(root) / file
            file_hash = get_file_hash(file_path)
            hash_sha256.update(file_hash.encode("utf-8"))

    return hash_sha256.hexdigest()
