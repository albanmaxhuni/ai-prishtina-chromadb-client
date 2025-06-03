"""
Additional features for Chroma vectorization and data processing.
"""

from typing import List, Dict, Any, Optional, Union, Tuple
import numpy as np
from dataclasses import dataclass
import chromadb
from chromadb.utils import embedding_functions
from chromadb.api.types import (
    Documents,
    Embeddings,
    Metadatas,
    Where,
    WhereDocument,
    QueryResult
)
from .exceptions import FeatureError

@dataclass
class FeatureConfig:
    """Configuration for feature extraction and processing."""
    normalize: bool = True
    dimensionality_reduction: Optional[int] = None
    feature_scaling: bool = True
    cache_features: bool = True
    batch_size: int = 100
    embedding_function: Optional[str] = "default"  # "default", "openai", "sentence_transformer"
    collection_name: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None
    persist_directory: Optional[str] = None
    collection_metadata: Optional[Dict[str, Any]] = None
    hnsw_config: Optional[Dict[str, Any]] = None
    distance_function: str = "cosine"  # "cosine", "l2", "ip"

class FeatureExtractor:
    """Base class for feature extraction."""
    
    def __init__(self, config: FeatureConfig):
        self.config = config
        self._feature_cache = {}
        self._setup_embedding_function()
    
    def _setup_embedding_function(self):
        """Setup the embedding function based on configuration."""
        if self.config.embedding_function == "openai":
            self.embedding_fn = embedding_functions.OpenAIEmbeddingFunction()
        elif self.config.embedding_function == "sentence_transformer":
            self.embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction()
        else:
            self.embedding_fn = embedding_functions.DefaultEmbeddingFunction()
    
    def extract(self, data: Any) -> np.ndarray:
        """Extract features from input data."""
        raise NotImplementedError
    
    def batch_extract(self, data_list: List[Any]) -> List[np.ndarray]:
        """Extract features from a batch of data."""
        return [self.extract(data) for data in data_list]

class TextFeatureExtractor(FeatureExtractor):
    """Extract features from text data using Chroma's embedding functions."""
    
    def __init__(self, config: FeatureConfig):
        super().__init__(config)
        self._text_features = {
            'embedding': self._extract_embedding,
            'length': self._extract_length,
            'complexity': self._extract_complexity
        }
    
    def extract(self, text: str) -> np.ndarray:
        """Extract features from text."""
        if self.config.cache_features and text in self._feature_cache:
            return self._feature_cache[text]
        
        features = []
        for feature_name, extractor in self._text_features.items():
            try:
                feature_value = extractor(text)
                if isinstance(feature_value, (list, np.ndarray)):
                    features.extend(feature_value)
                else:
                    features.append(feature_value)
            except Exception as e:
                print(f"Failed to extract {feature_name}: {str(e)}")
                if feature_name == 'embedding':
                    features.extend([0.0] * 1536)  # Default embedding size
                else:
                    features.append(0.0)
        
        features = np.array(features, dtype=np.float32)
        if self.config.normalize:
            features = self._normalize_features(features)
        
        if self.config.cache_features:
            self._feature_cache[text] = features
        
        return features

    def _extract_embedding(self, text: str) -> List[float]:
        """Extract text embedding using Chroma's embedding function."""
        try:
            embedding = self.embedding_fn([text])[0]
            return embedding
        except Exception as e:
            print(f"Failed to extract embedding: {str(e)}")
            return [0.0] * 1536  # Default embedding size

    def _extract_length(self, text: str) -> float:
        """Extract text length feature."""
        return float(len(text))

    def _extract_complexity(self, text: str) -> float:
        """Extract text complexity feature."""
        words = text.split()
        if not words:
            return 0.0
        return float(sum(len(word) for word in words) / len(words))

    def _normalize_features(self, features: np.ndarray) -> np.ndarray:
        """Normalize features to [0, 1] range."""
        if len(features) == 0:
            return features
        min_val = features.min()
        max_val = features.max()
        if max_val == min_val:
            return np.zeros_like(features)
        return (features - min_val) / (max_val - min_val)

class ImageFeatureExtractor(FeatureExtractor):
    """Extract features from image data."""
    
    def __init__(self, config: FeatureConfig):
        super().__init__(config)
        self._image_features = {
            'color': self._extract_color,
            'texture': self._extract_texture,
            'shape': self._extract_shape,
            'edges': self._extract_edges
        }
    
    def extract(self, image: Any) -> np.ndarray:
        """Extract features from image."""
        if self.config.cache_features and id(image) in self._feature_cache:
            return self._feature_cache[id(image)]
        
        features = []
        for feature_name, extractor in self._image_features.items():
            try:
                feature_value = extractor(image)
                features.append(feature_value)
            except Exception as e:
                print(f"Failed to extract {feature_name}: {str(e)}")
                features.append(0.0)
        
        features = np.array(features)
        if self.config.normalize:
            features = self._normalize_features(features)
        
        if self.config.cache_features:
            self._feature_cache[id(image)] = features
        
        return features
    
    @staticmethod
    def _extract_color(image: Any) -> np.ndarray:
        """Extract color features."""
        # Placeholder for color feature extraction
        return np.zeros(3)

    @staticmethod
    def _extract_texture(self, image: Any) -> np.ndarray:
        """Extract texture features."""
        # Placeholder for texture feature extraction
        return np.zeros(3)

    @staticmethod
    def _extract_shape(self, image: Any) -> np.ndarray:
        """Extract shape features."""
        # Placeholder for shape feature extraction
        return np.zeros(3)

    @staticmethod
    def _extract_edges(self, image: Any) -> np.ndarray:
        """Extract edge features."""
        # Placeholder for edge feature extraction
        return np.zeros(3)

    @staticmethod
    def _normalize_features(self, features: np.ndarray) -> np.ndarray:
        """Normalize features to [0, 1] range."""
        if len(features) == 0:
            return features
        return (features - features.min()) / (features.max() - features.min())

class AudioFeatureExtractor(FeatureExtractor):
    """Extract features from audio data."""
    
    def __init__(self, config: FeatureConfig):
        super().__init__(config)
        self._audio_features = {
            'mfcc': self._extract_mfcc,
            'spectral': self._extract_spectral,
            'temporal': self._extract_temporal,
            'rhythm': self._extract_rhythm
        }
    
    def extract(self, audio: Any) -> np.ndarray:
        """Extract features from audio."""
        if self.config.cache_features and id(audio) in self._feature_cache:
            return self._feature_cache[id(audio)]
        
        features = []
        for feature_name, extractor in self._audio_features.items():
            try:
                feature_value = extractor(audio)
                features.append(feature_value)
            except Exception as e:
                print(f"Failed to extract {feature_name}: {str(e)}")
                features.append(0.0)
        
        features = np.array(features)
        if self.config.normalize:
            features = self._normalize_features(features)
        
        if self.config.cache_features:
            self._feature_cache[id(audio)] = features
        
        return features

    @staticmethod
    def _extract_mfcc(self, audio: Any) -> np.ndarray:
        """Extract MFCC features."""
        # Placeholder for MFCC feature extraction
        return np.zeros(13)

    @staticmethod
    def _extract_spectral(self, audio: Any) -> np.ndarray:
        """Extract spectral features."""
        # Placeholder for spectral feature extraction
        return np.zeros(5)

    @staticmethod
    def _extract_temporal(self, audio: Any) -> np.ndarray:
        """Extract temporal features."""
        # Placeholder for temporal feature extraction
        return np.zeros(5)

    @staticmethod
    def _extract_rhythm(self, audio: Any) -> np.ndarray:
        """Extract rhythm features."""
        # Placeholder for rhythm feature extraction
        return np.zeros(5)

    @staticmethod
    def _normalize_features(self, features: np.ndarray) -> np.ndarray:
        """Normalize features to [0, 1] range."""
        if len(features) == 0:
            return features
        return (features - features.min()) / (features.max() - features.min())

class FeatureProcessor:
    """Process and manage features in ChromaDB collections."""
    
    def __init__(self, config: FeatureConfig):
        self.config = config
        self.client = None
        self.collection = None
        self._setup_chroma_client()
        self._setup_collection()
    
    def _setup_chroma_client(self):
        """Setup ChromaDB client."""
        try:
            settings = chromadb.Settings(
                anonymized_telemetry=False,
                allow_reset=True,
                persist_directory=self.config.persist_directory
            )
            if self.config.persist_directory:
                from chromadb import PersistentClient
                self.client = PersistentClient(path=self.config.persist_directory, settings=settings)
            else:
                self.client = chromadb.Client(settings)
        except Exception as e:
            print(f"Failed to initialize ChromaDB client: {str(e)}")
            self.client = None
    
    def _get_embedding_function(self):
        """Get the appropriate embedding function."""
        if self.config.embedding_function == "openai":
            return embedding_functions.OpenAIEmbeddingFunction()
        elif self.config.embedding_function == "sentence_transformer":
            return embedding_functions.SentenceTransformerEmbeddingFunction()
        else:
            return embedding_functions.DefaultEmbeddingFunction()
    
    def _setup_collection(self):
        """Setup or get existing collection."""
        if not self.client:
            raise FeatureError("Failed to initialize ChromaDB client")
            
        if not self.config.collection_name:
            raise FeatureError("No collection name configured")
            
        try:
            embedding_function = self._get_embedding_function()
            self.collection = self.client.get_or_create_collection(
                name=self.config.collection_name,
                metadata=self.config.collection_metadata,
                embedding_function=embedding_function
            )
        except Exception as e:
            raise FeatureError(f"Failed to get/create collection: {str(e)}")
    
    def __del__(self):
        """Cleanup when the processor is destroyed."""
        if self.client:
            try:
                self.client.reset()
            except:
                pass
    
    def process(self, data: Dict[str, Any]) -> np.ndarray:
        """Process input data and return feature vector."""
        if not data:
            raise FeatureError("Empty data provided")
            
        features = []
        for data_type, value in data.items():
            if data_type == "text":
                extractor = TextFeatureExtractor(self.config)
                feature_vector = extractor.extract(value)
                features.append(feature_vector)
            elif data_type == "image":
                extractor = ImageFeatureExtractor(self.config)
                feature_vector = extractor.extract(value)
                features.append(feature_vector)
            elif data_type == "audio":
                extractor = AudioFeatureExtractor(self.config)
                feature_vector = extractor.extract(value)
                features.append(feature_vector)
            else:
                raise FeatureError(f"Unsupported data type: {data_type}")
        
        if not features:
            raise FeatureError("No features extracted")
            
        # Combine features
        combined_features = np.concatenate(features)
        
        # Apply dimensionality reduction if configured
        if self.config.dimensionality_reduction:
            combined_features = self._reduce_dimensions(combined_features)
            
        return combined_features
    
    def _reduce_dimensions(self, features: np.ndarray) -> np.ndarray:
        """Reduce feature dimensions if configured."""
        if not self.config.dimensionality_reduction:
            return features
            
        target_dim = self.config.dimensionality_reduction
        if len(features) <= target_dim:
            return features
            
        # Simple PCA-like reduction
        mean = np.mean(features, axis=0)
        centered = features - mean
        cov = np.cov(centered.T)
        eigenvals, eigenvecs = np.linalg.eigh(cov)
        idx = eigenvals.argsort()[::-1]
        eigenvecs = eigenvecs[:, idx]
        return np.dot(centered, eigenvecs[:, :target_dim])
    
    def add_to_collection(
        self,
        data: Dict[str, Any],
        id: str,
        metadata: Optional[Dict[str, Any]] = None,
        documents: Optional[List[str]] = None
    ):
        """Add processed features to Chroma collection."""
        if not self.collection:
            raise FeatureError("No collection configured")
        
        features = self.process(data)
        self.collection.add(
            embeddings=[features.tolist()],
            ids=[id],
            metadatas=[metadata or self.config.metadata or {}],
            documents=documents
        )
    
    def query_collection(
        self,
        query_data: Dict[str, Any],
        n_results: int = 5,
        where: Optional[Where] = None,
        where_document: Optional[WhereDocument] = None,
        include: Optional[List[str]] = None
    ) -> QueryResult:
        """Query the collection with processed features."""
        if not self.collection:
            raise FeatureError("No collection configured")
        
        query_features = self.process(query_data)
        if include is None:
            include = ["embeddings", "metadatas", "documents"]
        results = self.collection.query(
            query_embeddings=[query_features.tolist()],
            n_results=n_results,
            where=where,
            where_document=where_document,
            include=include
        )
        return results
    
    def update_collection(
        self,
        ids: List[str],
        embeddings: Optional[List[List[float]]] = None,
        metadatas: Optional[List[Dict[str, Any]]] = None,
        documents: Optional[List[str]] = None
    ):
        """Update items in the collection."""
        if not self.collection:
            raise FeatureError("No collection configured")
        
        self.collection.update(
            ids=ids,
            embeddings=embeddings,
            metadatas=metadatas,
            documents=documents
        )
    
    def delete_from_collection(
        self,
        ids: Optional[List[str]] = None,
        where: Optional[Where] = None,
        where_document: Optional[WhereDocument] = None
    ):
        """Delete items from the collection."""
        if not self.collection:
            raise FeatureError("No collection configured")
        
        self.collection.delete(
            ids=ids,
            where=where,
            where_document=where_document
        )
    
    def get_collection_stats(self) -> Dict[str, Any]:
        """Get collection statistics."""
        if not self.collection:
            raise FeatureError("No collection configured")
        
        return {
            "count": self.collection.count(),
            "name": self.collection.name,
            "metadata": self.collection.metadata
        }

class FeatureRegistry:
    """Registry for managing feature extractors and processors."""
    
    def __init__(self):
        self._extractors = {}
        self._processors = {}
    
    def register_extractor(self, name: str, extractor: FeatureExtractor):
        """Register a feature extractor."""
        self._extractors[name] = extractor
    
    def register_processor(self, name: str, processor: FeatureProcessor):
        """Register a feature processor."""
        self._processors[name] = processor
    
    def get_extractor(self, name: str) -> FeatureExtractor:
        """Get a registered feature extractor."""
        if name not in self._extractors:
            raise FeatureError(f"Extractor '{name}' not found")
        return self._extractors[name]
    
    def get_processor(self, name: str) -> FeatureProcessor:
        """Get a registered feature processor."""
        if name not in self._processors:
            raise FeatureError(f"Processor '{name}' not found")
        return self._processors[name] 