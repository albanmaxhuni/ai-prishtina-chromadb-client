"""
Base API client for AIPrishtina VectorDB.
"""

from typing import Dict, Any, Optional, Union
import requests
from .exceptions import (
    APIError,
    APIConfigurationError,
    APIAuthenticationError,
    APIRateLimitError,
    APIValidationError,
    APINotFoundError,
    APIConnectionError,
    APITimeoutError,
    APIServerError,
    APIClientError
)

class BaseAPIClient:
    """Base class for API clients."""
    
    def __init__(
        self,
        base_url: str,
        api_key: Optional[str] = None,
        timeout: int = 30,
        max_retries: int = 3,
        headers: Optional[Dict[str, str]] = None
    ):
        """Initialize the API client.
        
        Args:
            base_url: Base URL for the API
            api_key: API key for authentication
            timeout: Request timeout in seconds
            max_retries: Maximum number of retries for failed requests
            headers: Additional headers to include in requests
        """
        self.base_url = base_url.rstrip('/')
        self.api_key = api_key
        self.timeout = timeout
        self.max_retries = max_retries
        self.headers = headers or {}
        
        if api_key:
            self.headers['Authorization'] = f'Bearer {api_key}'
        
        self.session = requests.Session()
        self.session.headers.update(self.headers)
    
    def _make_request(
        self,
        method: str,
        endpoint: str,
        params: Optional[Dict[str, Any]] = None,
        data: Optional[Dict[str, Any]] = None,
        json: Optional[Dict[str, Any]] = None,
        headers: Optional[Dict[str, str]] = None
    ) -> Dict[str, Any]:
        """Make an HTTP request to the API.
        
        Args:
            method: HTTP method (GET, POST, etc.)
            endpoint: API endpoint
            params: Query parameters
            data: Form data
            json: JSON data
            headers: Additional headers
            
        Returns:
            Response data as dictionary
            
        Raises:
            APIError: If the request fails
        """
        url = f"{self.base_url}/{endpoint.lstrip('/')}"
        
        try:
            response = self.session.request(
                method=method,
                url=url,
                params=params,
                data=data,
                json=json,
                headers=headers,
                timeout=self.timeout
            )
            
            # Handle different status codes
            if response.status_code == 200:
                return response.json()
            elif response.status_code == 400:
                raise APIValidationError(f"Validation error: {response.text}")
            elif response.status_code == 401:
                raise APIAuthenticationError(f"Authentication error: {response.text}")
            elif response.status_code == 403:
                raise APIAuthenticationError(f"Permission denied: {response.text}")
            elif response.status_code == 404:
                raise APINotFoundError(f"Resource not found: {response.text}")
            elif response.status_code == 429:
                raise APIRateLimitError(f"Rate limit exceeded: {response.text}")
            elif response.status_code >= 500:
                raise APIServerError(f"Server error: {response.text}")
            else:
                raise APIError(f"Unexpected error: {response.text}")
                
        except requests.exceptions.Timeout:
            raise APITimeoutError("Request timed out")
        except requests.exceptions.ConnectionError:
            raise APIConnectionError("Failed to connect to the API")
        except requests.exceptions.RequestException as e:
            raise APIClientError(f"Request failed: {str(e)}")
    
    def get(self, endpoint: str, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Make a GET request.
        
        Args:
            endpoint: API endpoint
            params: Query parameters
            
        Returns:
            Response data as dictionary
        """
        return self._make_request('GET', endpoint, params=params)
    
    def post(
        self,
        endpoint: str,
        data: Optional[Dict[str, Any]] = None,
        json: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Make a POST request.
        
        Args:
            endpoint: API endpoint
            data: Form data
            json: JSON data
            
        Returns:
            Response data as dictionary
        """
        return self._make_request('POST', endpoint, data=data, json=json)
    
    def put(
        self,
        endpoint: str,
        data: Optional[Dict[str, Any]] = None,
        json: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Make a PUT request.
        
        Args:
            endpoint: API endpoint
            data: Form data
            json: JSON data
            
        Returns:
            Response data as dictionary
        """
        return self._make_request('PUT', endpoint, data=data, json=json)
    
    def delete(self, endpoint: str) -> Dict[str, Any]:
        """Make a DELETE request.
        
        Args:
            endpoint: API endpoint
            
        Returns:
            Response data as dictionary
        """
        return self._make_request('DELETE', endpoint)
    
    def close(self):
        """Close the session."""
        self.session.close() 