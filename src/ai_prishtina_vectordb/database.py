"""
Vector database functionality for the AIPrishtina VectorDB library.
"""

from typing import Any, Dict, List, Optional, Union
import numpy as np
import chromadb
from chromadb.utils import embedding_functions
from .data_sources import DataSource
from .validation import (
    validate_metadata,
    validate_documents,
    validate_embeddings,
    validate_query_params,
    validate_index_params,
    validate_ids
)
from .exceptions import DatabaseError, ValidationError
import os
from pathlib import Path
from .config import DatabaseConfig
from .logger import AIPrishtinaLogger
from datetime import datetime
import json
import tempfile

class Database:
    """
    A professional vector database for storing and querying vector embeddings.
    
    This class provides a unified interface for vector storage and similarity search,
    supporting various data sources and indexing methods.
    """
    
    def __init__(
        self,
        collection_name: str = "default",
        config: Optional[Dict[str, Any]] = None
    ):
        """Initialize database.
        
        Args:
            collection_name: Name of the collection
            config: Configuration dictionary
        """
        self.logger = AIPrishtinaLogger()
        self.config = config or {}
        self.collection_name = collection_name
        
        # Initialize ChromaDB client with minimal settings
        try:
            import chromadb
            from chromadb.utils import embedding_functions
            
            # Create a temporary directory for persistence
            persist_dir = tempfile.mkdtemp()
            
            # Initialize client using new API
            self.client = chromadb.PersistentClient(path=persist_dir)
            
            # Use a simple embedding function for testing that doesn't require model download
            embedding_function = embedding_functions.DefaultEmbeddingFunction()
            
            # Create or get collection with minimal metadata
            self.collection = self.client.get_or_create_collection(
                name=self.collection_name,
                embedding_function=embedding_function,
                metadata={
                    "embedding_model": "default",
                    "index_type": "default"
                }
            )
            self.logger.info(f"Initialized database with collection: {collection_name}")
        except Exception as e:
            self.logger.error(f"Failed to initialize database: {str(e)}")
            raise DatabaseError(f"Failed to initialize database: {str(e)}")

    def add_from_source(
        self,
        source: Union[str, List[Dict[str, Any]]],
        source_type: str = "text",
        text_column: Optional[str] = None,
        metadata_columns: Optional[List[str]] = None,
        **kwargs
    ) -> None:
        """
        Add data from various sources to the database.
        
        Args:
            source: Data source (file path, DataFrame, or list of dicts)
            source_type: Type of data source ('text', 'json', 'csv', 'pandas')
            text_column: Column name containing text to vectorize
            metadata_columns: Columns to include as metadata
            **kwargs: Additional parameters for data loading
        """
        try:
            data_source = DataSource(source_type=source_type, **kwargs)
            data = data_source.load_data(
                source=source,
                text_column=text_column,
                metadata_columns=metadata_columns,
                **kwargs
            )
            
            # Validate data
            validate_documents(data['documents'])
            for metadata in data['metadatas']:
                validate_metadata(metadata)
            
            self.collection.add(
                documents=data['documents'],
                metadatas=data['metadatas'],
                ids=data['ids']
            )
            self.logger.info(f"Added {len(data['documents'])} documents to collection")
        except Exception as e:
            raise DatabaseError(f"Failed to add data from source: {str(e)}")

    def add(
        self,
        embeddings: Optional[np.ndarray] = None,
        documents: Optional[List[str]] = None,
        metadatas: Optional[List[Dict[str, Any]]] = None,
        ids: Optional[List[str]] = None
    ) -> None:
        """
        Add vectors to the database.
        
        Args:
            embeddings: Optional vector embeddings to add
            documents: Optional documents associated with the embeddings
            metadatas: Optional metadata for each embedding
            ids: Optional IDs for each embedding
        """
        try:
            if embeddings is None and documents is None:
                raise ValidationError("Either embeddings or documents must be provided")

            if ids is None:
                length = len(documents) if documents is not None else len(embeddings)
                ids = [str(i) for i in range(length)]

            if metadatas is None:
                metadatas = [{"type": "default"} for _ in range(len(ids))]

            # Validate inputs
            if documents is not None:
                validate_documents(documents)
            if embeddings is not None:
                validate_embeddings(embeddings)
            for metadata in metadatas:
                validate_metadata(metadata)

            if embeddings is None and documents is not None:
                # Generate embeddings from documents
                embeddings = self.collection._embedding_function(documents)

            self.collection.add(
                embeddings=embeddings.tolist() if isinstance(embeddings, np.ndarray) else embeddings,
                documents=documents,
                metadatas=metadatas,
                ids=ids
            )
            self.logger.info(f"Added {len(documents) if documents is not None else len(embeddings)} vectors to collection")
        except Exception as e:
            raise DatabaseError(f"Failed to add vectors: {str(e)}")

    def query(
        self,
        query_texts: Optional[List[str]] = None,
        query_embeddings: Optional[np.ndarray] = None,
        n_results: int = 10,
        where: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Query the database for similar vectors.
        
        Args:
            query_texts: Optional query texts
            query_embeddings: Optional query vector embeddings
            n_results: Number of results to return
            where: Optional filter conditions
            **kwargs: Additional query parameters
            
        Returns:
            Dict containing query results
        """
        try:
            # Validate query parameters
            validate_query_params(
                query_texts=query_texts,
                query_embeddings=query_embeddings,
                n_results=n_results,
                where=where
            )

            if query_texts is not None:
                results = self.collection.query(
                    query_texts=query_texts,
                    n_results=n_results,
                    where=where,
                    **kwargs
                )
            else:
                results = self.collection.query(
                    query_embeddings=query_embeddings.tolist(),
                    n_results=n_results,
                    where=where,
                    **kwargs
                )
                
            self.logger.info(f"Queried {len(query_texts) if query_texts else len(query_embeddings)} vectors")
            return results
        except Exception as e:
            raise DatabaseError(f"Failed to query database: {str(e)}")

    def delete(
        self,
        ids: Optional[List[str]] = None,
        where: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Delete vectors from the database.
        
        Args:
            ids: Optional list of IDs to delete
            where: Optional filter conditions for deletion
        """
        try:
            if where is not None:
                validate_metadata(where)
            
            self.collection.delete(
                ids=ids,
                where=where
            )
            self.logger.info("Deleted vectors from collection")
        except Exception as e:
            raise DatabaseError(f"Failed to delete vectors: {str(e)}")

    def get(
        self,
        ids: Optional[List[str]] = None,
        where: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Get vectors from the database.
        
        Args:
            ids: Optional list of IDs to get
            where: Optional filter conditions
            **kwargs: Additional get parameters
            
        Returns:
            Dict containing the requested vectors
        """
        try:
            if where is not None:
                validate_metadata(where)
            
            results = self.collection.get(
                ids=ids,
                where=where,
                **kwargs
            )
            self.logger.info("Retrieved vectors from collection")
            return results
        except Exception as e:
            raise DatabaseError(f"Failed to get vectors: {str(e)}")

    def update(
        self,
        ids: List[str],
        embeddings: Optional[np.ndarray] = None,
        documents: Optional[List[str]] = None,
        metadatas: Optional[List[Dict[str, Any]]] = None
    ) -> None:
        """
        Update vectors in the database.
        
        Args:
            ids: List of IDs to update
            embeddings: Optional new embeddings
            documents: Optional new documents
            metadatas: Optional new metadata
        """
        try:
            # Validate inputs
            if documents is not None:
                validate_documents(documents)
            if embeddings is not None:
                validate_embeddings(embeddings)
            if metadatas is not None:
                for metadata in metadatas:
                    validate_metadata(metadata)
            
            # Convert embeddings to list if provided
            embeddings_list = None
            if embeddings is not None:
                if isinstance(embeddings, np.ndarray):
                    embeddings_list = embeddings.tolist()
                else:
                    embeddings_list = embeddings
            
            self.collection.update(
                ids=ids,
                embeddings=embeddings_list,
                documents=documents,
                metadatas=metadatas
            )
            self.logger.info(f"Updated {len(ids)} vectors")
        except Exception as e:
            raise DatabaseError(f"Failed to update vectors: {str(e)}")

    def create_index(
        self,
        index_type: str = "hnsw",
        **kwargs
    ) -> None:
        """
        Create an index for efficient similarity search.
        
        Args:
            index_type: Type of index to create ('hnsw', 'flat', etc.)
            **kwargs: Additional index parameters
        """
        try:
            # Validate index parameters
            validate_index_params(index_type, kwargs)
            
            self.collection.create_index(
                index_type=index_type,
                **kwargs
            )
            self.logger.info("Created collection index")
        except Exception as e:
            raise DatabaseError(f"Failed to create index: {str(e)}")

    def delete_collection(self) -> None:
        """Delete the collection."""
        try:
            self.client.delete_collection(self.collection.name)
            self.logger.info(f"Deleted collection: {self.collection.name}")
        except Exception as e:
            raise DatabaseError(f"Failed to delete collection: {str(e)}") 