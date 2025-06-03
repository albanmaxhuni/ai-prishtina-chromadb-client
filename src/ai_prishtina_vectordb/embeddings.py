"""
Embedding functionality for AIPrishtina VectorDB.
"""

import numpy as np
from typing import List, Union, Optional
from .logger import AIPrishtinaLogger
from .exceptions import EmbeddingError
import os

class EmbeddingModel:
    """Model for generating embeddings."""
    
    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        hf_token: Optional[str] = None,
        device: str = "cpu"
    ):
        """Initialize embedding model.
        
        Args:
            model_name: Name of the model to use
            hf_token: Hugging Face API token
            device: Device to use for model inference (cpu/cuda)
        """
        self.model_name = model_name
        self.hf_token = hf_token
        self.device = device
        self.logger = AIPrishtinaLogger()
        self._init_model()
        self.dimension = self.get_embedding_dimension()
        
    def _init_model(self) -> None:
        """Initialize the embedding model."""
        try:
            import sentence_transformers
            from huggingface_hub import HfFolder
            import requests
            from requests.adapters import HTTPAdapter
            from urllib3.util.retry import Retry

            # Configure retry strategy
            retry_strategy = Retry(
                total=3,
                backoff_factor=1,
                status_forcelist=[429, 500, 502, 503, 504],
            )
            adapter = HTTPAdapter(max_retries=retry_strategy)
            session = requests.Session()
            session.mount("https://", adapter)
            session.mount("http://", adapter)

            # Set custom session for huggingface_hub
            if hasattr(self, 'hf_token') and self.hf_token:
                HfFolder.save_token(self.hf_token)
            self.model = sentence_transformers.SentenceTransformer(
                self.model_name,
                cache_folder=os.path.join(os.path.expanduser("~"), ".cache", "huggingface"),
                use_auth_token=False,
                device=self.device
            )
            self.logger.info(f"Initialized embedding model: {self.model_name}")
        except requests.exceptions.ConnectionError as e:
            self.logger.error(f"Network error while initializing model: {str(e)}")
            raise EmbeddingError(f"Failed to connect to Hugging Face Hub. Please check your internet connection and DNS settings.")
        except Exception as e:
            self.logger.error(f"Failed to initialize embedding model: {str(e)}")
            raise EmbeddingError(f"Failed to initialize embedding model: {str(e)}")
            
    def encode(self, texts: List[str], batch_size: int = 32, **kwargs) -> np.ndarray:
        """Generate embeddings for texts.
        
        Args:
            texts: List of text strings
            batch_size: Batch size for processing
            **kwargs: Additional parameters for the model
            
        Returns:
            Array of embeddings
        """
        try:
            embeddings = self.model.encode(texts, batch_size=batch_size, **kwargs)
            self.logger.debug(f"Generated embeddings for {len(texts)} texts")
            return embeddings
        except Exception as e:
            raise EmbeddingError(f"Failed to generate embeddings: {str(e)}")
            
    def embed_text(self, texts: List[str]) -> np.ndarray:
        """Generate embeddings for text.
        
        Args:
            texts: List of text strings
            
        Returns:
            Array of embeddings
        """
        return self.encode(texts)
            
    def embed_image(self, images: np.ndarray) -> np.ndarray:
        """Generate embeddings for images.
        
        Args:
            images: Array of images
            
        Returns:
            Array of embeddings
        """
        try:
            # Convert images to list of PIL Images
            from PIL import Image
            import torch
            from torchvision import transforms
            from torchvision.models import resnet50, ResNet50_Weights
            
            # Initialize ResNet model
            model = resnet50(weights=ResNet50_Weights.DEFAULT)
            model = torch.nn.Sequential(*list(model.children())[:-1])  # Remove last layer
            model.eval()
            
            # Define image transform
            transform = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
            
            # Process images
            embeddings = []
            for img in images:
                # Convert float32/float64 to uint8
                if img.dtype in [np.float32, np.float64]:
                    img = (img * 255).astype(np.uint8)
                pil_img = Image.fromarray(img)
                tensor = transform(pil_img).unsqueeze(0)
                
                # Generate embedding
                with torch.no_grad():
                    embedding = model(tensor).squeeze().numpy()
                embeddings.append(embedding)
            
            embeddings = np.array(embeddings)
            self.logger.debug(f"Generated embeddings for {len(images)} images")
            return embeddings
        except Exception as e:
            raise EmbeddingError(f"Failed to generate image embeddings: {str(e)}")
            
    def embed_audio(self, audio: np.ndarray) -> np.ndarray:
        """Generate embeddings for audio.
        
        Args:
            audio: Array of audio data
            
        Returns:
            Array of embeddings
        """
        try:
            # Convert audio to text descriptions for now
            # In a real implementation, you would use a proper audio embedding model
            audio_descriptions = [f"Audio sample {i}" for i in range(len(audio))]
            embeddings = self.encode(audio_descriptions)
            self.logger.debug(f"Generated embeddings for {len(audio)} audio samples")
            return embeddings
        except Exception as e:
            raise EmbeddingError(f"Failed to generate audio embeddings: {str(e)}")
            
    def embed_video(self, video: np.ndarray) -> np.ndarray:
        """Generate embeddings for video.
        
        Args:
            video: Array of video frames
            
        Returns:
            Array of embeddings
        """
        try:
            # Convert video frames to list of PIL Images
            from PIL import Image
            import torch
            from torchvision import transforms
            from torchvision.models import resnet50, ResNet50_Weights
            
            # Initialize ResNet model
            model = resnet50(weights=ResNet50_Weights.DEFAULT)
            model = torch.nn.Sequential(*list(model.children())[:-1])  # Remove last layer
            model.eval()
            
            # Define image transform
            transform = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
            
            # Process video frames
            embeddings = []
            for v in video:
                frame_embeddings = []
                for frame in v:
                    # Convert float32/float64 to uint8
                    if frame.dtype in [np.float32, np.float64]:
                        frame = (frame * 255).astype(np.uint8)
                    pil_frame = Image.fromarray(frame)
                    tensor = transform(pil_frame).unsqueeze(0)
                    
                    # Generate embedding
                    with torch.no_grad():
                        embedding = model(tensor).squeeze().numpy()
                    frame_embeddings.append(embedding)
                
                # Average frame embeddings for the video
                video_embedding = np.mean(frame_embeddings, axis=0)
                embeddings.append(video_embedding)
            
            embeddings = np.array(embeddings)
            self.logger.debug(f"Generated embeddings for {len(video)} videos")
            return embeddings
        except Exception as e:
            raise EmbeddingError(f"Failed to generate video embeddings: {str(e)}")
            
    def normalize_embeddings(self, embeddings: np.ndarray) -> np.ndarray:
        """Normalize embeddings to unit length.
        
        Args:
            embeddings: Array of embeddings
            
        Returns:
            Normalized embeddings
        """
        try:
            norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
            normalized = embeddings / norms
            self.logger.debug("Normalized embeddings")
            return normalized
        except Exception as e:
            raise EmbeddingError(f"Failed to normalize embeddings: {str(e)}")
            
    def get_embedding_dimension(self) -> int:
        """Get the dimension of the embeddings.
        
        Returns:
            Embedding dimension
        """
        try:
            return self.model.get_sentence_embedding_dimension()
        except Exception as e:
            raise EmbeddingError(f"Failed to get embedding dimension: {str(e)}") 