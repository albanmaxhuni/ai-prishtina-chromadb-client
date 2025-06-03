"""
Validation utilities for AI Prishtina VectorDB.
"""

from typing import Any, Dict, List, Optional, Union
import numpy as np
from .exceptions import ValidationError

def validate_metadata(metadata: Union[Dict[str, Any], List[Dict[str, Any]]]) -> bool:
    """Validate metadata.
    
    Args:
        metadata: Metadata dictionary or list of metadata dictionaries
        
    Returns:
        bool: True if validation passes
        
    Raises:
        ValidationError: If metadata is invalid
    """
    if isinstance(metadata, dict):
        if not all(isinstance(k, str) for k in metadata.keys()):
            raise ValidationError("Metadata keys must be strings")
        if not all(isinstance(v, (str, int, float, bool)) for v in metadata.values()):
            raise ValidationError("Metadata values must be strings, numbers, or booleans")
    elif isinstance(metadata, list):
        if not all(isinstance(m, dict) for m in metadata):
            raise ValidationError("All metadata items must be dictionaries")
        for m in metadata:
            validate_metadata(m)
    else:
        raise ValidationError("Metadata must be a dictionary or list of dictionaries")
    return True

def validate_documents(
    documents: List[str],
    metadatas: Optional[List[Dict[str, Any]]] = None,
    ids: Optional[List[str]] = None
) -> bool:
    """Validate documents and associated metadata/IDs.
    
    Args:
        documents: List of document texts
        metadatas: Optional list of metadata dictionaries
        ids: Optional list of document IDs
        
    Returns:
        bool: True if validation passes
        
    Raises:
        ValidationError: If validation fails
    """
    if not isinstance(documents, list):
        raise ValidationError("Documents must be a list")
    if not all(isinstance(d, str) for d in documents):
        raise ValidationError("All documents must be strings")
        
    if metadatas is not None:
        if not isinstance(metadatas, list):
            raise ValidationError("Metadatas must be a list")
        if len(metadatas) != len(documents):
            raise ValidationError("Number of metadatas must match number of documents")
        validate_metadata(metadatas)
        
    if ids is not None:
        if not isinstance(ids, list):
            raise ValidationError("IDs must be a list")
        if len(ids) != len(documents):
            raise ValidationError("Number of IDs must match number of documents")
        if not all(isinstance(i, str) for i in ids):
            raise ValidationError("All IDs must be strings")
        if len(set(ids)) != len(ids):
            raise ValidationError("IDs must be unique")
    return True

def validate_embeddings(embeddings: np.ndarray) -> None:
    """Validate embeddings array."""
    if not isinstance(embeddings, np.ndarray):
        raise ValidationError("Embeddings must be a numpy array")
    
    if len(embeddings.shape) != 2:
        raise ValidationError("Embeddings must be a 2D array")
    
    if np.isnan(embeddings).any():
        raise ValidationError("Embeddings cannot contain NaN values")

def validate_query_params(
    query_texts: Optional[List[str]] = None,
    query_embeddings: Optional[np.ndarray] = None,
    n_results: int = 5,
    where: Optional[Dict[str, Any]] = None
) -> None:
    """Validate query parameters."""
    if query_texts is None and query_embeddings is None:
        raise ValidationError("Either query_texts or query_embeddings must be provided")
    
    if query_texts is not None:
        validate_documents(query_texts)
    
    if query_embeddings is not None:
        validate_embeddings(query_embeddings)
    
    if not isinstance(n_results, int) or n_results <= 0:
        raise ValidationError("n_results must be a positive integer")
    
    if where is not None:
        validate_metadata(where)

def validate_batch_params(
    batch_size: int,
    max_workers: int
) -> None:
    """Validate batch processing parameters."""
    if not isinstance(batch_size, int) or batch_size <= 0:
        raise ValidationError("batch_size must be a positive integer")
    
    if not isinstance(max_workers, int) or max_workers <= 0:
        raise ValidationError("max_workers must be a positive integer")

def validate_index_params(
    index_type: str,
    params: Dict[str, Any]
) -> None:
    """Validate index parameters."""
    valid_types = {"hnsw", "ivf", "flat"}
    if index_type not in valid_types:
        raise ValidationError(f"Invalid index type. Must be one of: {valid_types}")
    
    if not isinstance(params, dict):
        raise ValidationError("Index parameters must be a dictionary")
    
    if index_type == "hnsw":
        required_params = {"M", "ef_construction", "ef_search"}
        missing = required_params - set(params.keys())
        if missing:
            raise ValidationError(f"Missing required HNSW parameters: {missing}")
    
    elif index_type == "ivf":
        required_params = {"nlist", "nprobe"}
        missing = required_params - set(params.keys())
        if missing:
            raise ValidationError(f"Missing required IVF parameters: {missing}")

def validate_ids(ids: List[str]) -> bool:
    """Validate document IDs.
    
    Args:
        ids: List of document IDs
        
    Returns:
        bool: True if validation passes
        
    Raises:
        ValidationError: If validation fails
    """
    if not isinstance(ids, list):
        raise ValidationError("IDs must be a list")
    if not all(isinstance(i, str) for i in ids):
        raise ValidationError("All IDs must be strings")
    if not all(i.strip() for i in ids):
        raise ValidationError("IDs cannot be empty strings")
    if len(set(ids)) != len(ids):
        raise ValidationError("IDs must be unique")
    return True

def validate_lengths(
    documents: List[str],
    metadatas: Optional[List[Dict[str, Any]]] = None,
    ids: Optional[List[str]] = None
) -> bool:
    """Validate lengths of documents, metadatas, and IDs.
    
    Args:
        documents: List of document texts
        metadatas: Optional list of metadata dictionaries
        ids: Optional list of document IDs
        
    Returns:
        bool: True if validation passes
        
    Raises:
        ValidationError: If lengths don't match
    """
    if metadatas is not None and len(metadatas) != len(documents):
        raise ValidationError("Number of metadatas must match number of documents")
    if ids is not None and len(ids) != len(documents):
        raise ValidationError("Number of IDs must match number of documents")
    return True 