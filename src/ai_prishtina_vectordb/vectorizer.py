"""
Core vectorization functionality for the AIPrishtina VectorDB library.
"""

from typing import Any, Dict, List, Optional, Union
import numpy as np
import pandas as pd
from .embeddings import EmbeddingModel
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.feature_extraction import FeatureHasher

class Vectorizer:
    """
    A professional vectorizer for converting various data types into vector embeddings.
    
    This class provides methods to vectorize different types of data including:
    - Text data
    - Numerical data
    - Categorical data
    - Mixed data types
    """
    
    def __init__(
        self,
        embedding_model: Optional[EmbeddingModel] = None,
        normalize: bool = True,
        **kwargs
    ):
        """
        Initialize the Vectorizer.
        
        Args:
            embedding_model: Optional custom embedding model
            normalize: Whether to normalize vectors
            **kwargs: Additional configuration parameters
        """
        self.embedding_model = embedding_model or EmbeddingModel()
        self.normalize = normalize
        self.config = kwargs

    def vectorize_text(
        self,
        texts: Union[str, List[str]],
        batch_size: int = 32,
        **kwargs
    ) -> np.ndarray:
        """
        Vectorize text data using the embedding model.
        
        Args:
            texts: Single text or list of texts to vectorize
            batch_size: Batch size for processing
            **kwargs: Additional parameters for the embedding model
            
        Returns:
            numpy.ndarray: Vector embeddings
        """
        if isinstance(texts, str):
            texts = [texts]
            
        embeddings = self.embedding_model.encode(texts, batch_size=batch_size, **kwargs)
        
        if self.normalize:
            embeddings = self._normalize_vectors(embeddings)
            
        return embeddings

    def vectorize_numerical(
        self,
        data: Union[np.ndarray, pd.DataFrame],
        method: str = "standard",
        **kwargs
    ) -> np.ndarray:
        """
        Vectorize numerical data using various methods.
        
        Args:
            data: Numerical data to vectorize
            method: Vectorization method ('standard', 'minmax', 'robust')
            **kwargs: Additional parameters for the vectorization method
            
        Returns:
            numpy.ndarray: Vector embeddings
        """
        if isinstance(data, pd.DataFrame):
            data = data.values
            
        if method == "standard":
            from sklearn.preprocessing import StandardScaler
            scaler = StandardScaler(**kwargs)
        elif method == "minmax":
            from sklearn.preprocessing import MinMaxScaler
            scaler = MinMaxScaler(**kwargs)
        elif method == "robust":
            from sklearn.preprocessing import RobustScaler
            scaler = RobustScaler(**kwargs)
        else:
            raise ValueError(f"Unknown method: {method}")
            
        vectors = scaler.fit_transform(data)
        
        if self.normalize:
            vectors = self._normalize_vectors(vectors)
            
        return vectors

    def vectorize_categorical(
        self,
        data: Union[List[str], np.ndarray],
        method: str = "onehot",
        **kwargs
    ) -> np.ndarray:
        """
        Vectorize categorical data.

        Args:
            data: Categorical data to vectorize
            method: Vectorization method ('onehot', 'label', or 'hash')
            **kwargs: Additional parameters for the vectorization method

        Returns:
            Vectorized data as numpy array
        """
        if method == "onehot":
            encoder = OneHotEncoder(sparse_output=False, **kwargs)
            data_array = np.array(data).reshape(-1, 1)
            vectors = encoder.fit_transform(data_array)
        elif method == "label":
            encoder = LabelEncoder()
            vectors = encoder.fit_transform(data).reshape(-1, 1)
        elif method == "hash":
            hasher = FeatureHasher(n_features=kwargs.get("n_features", 10))
            vectors = hasher.transform([{str(i): x} for i, x in enumerate(data)]).toarray()
        else:
            raise ValueError(f"Unknown vectorization method: {method}")

        return vectors

    def _normalize_vectors(self, vectors: np.ndarray) -> np.ndarray:
        """
        Normalize vectors to unit length.
        
        Args:
            vectors: Input vectors
            
        Returns:
            numpy.ndarray: Normalized vectors
        """
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        return vectors / (norms + 1e-8)  # Add small epsilon to avoid division by zero 