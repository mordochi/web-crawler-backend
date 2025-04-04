"""
LLM Provider Integration Module for Web Crawler Backend

This module provides a unified interface for different LLM providers:
- OpenAI
- Claude (Anthropic)
- Ollama (local LLM)

It reads configuration from config.toml and environment variables.
"""

import os
import tomli
from typing import Dict, Any, Optional, Literal
from pathlib import Path
from dotenv import load_dotenv
import requests
import json
import logging

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Provider types
ProviderType = Literal["openai", "claude", "ollama"]

class LLMProviderConfig:
    """Configuration for LLM providers"""
    
    def __init__(self):
        self.config = self._load_config()
        self.default_provider = self._get_default_provider()
        
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from config.toml file"""
        config_path = Path(__file__).parent / "config.toml"
        
        if not config_path.exists():
            logger.warning(f"Config file not found at {config_path}. Using default configuration.")
            return {
                "llm": {
                    "default_provider": "openai",
                    "openai": {
                        "model": "gpt-3.5-turbo",
                        "temperature": 0.1
                    },
                    "claude": {
                        "model": "claude-3-sonnet-20240229",
                        "temperature": 0.0,
                        "base_url": "https://api.anthropic.com/v1/"
                    },
                    "ollama": {
                        "model": "llama3.2:latest",
                        "temperature": 0.0,
                        "base_url": "http://localhost:11434/v1"
                    }
                }
            }
        
        try:
            with open(config_path, "rb") as f:
                return tomli.load(f)
        except Exception as e:
            logger.error(f"Error loading config file: {e}")
            return {}
    
    def _get_default_provider(self) -> ProviderType:
        """Get the default LLM provider from environment or config"""
        # Check environment variable first
        env_provider = os.getenv("LLM_PROVIDER")
        if env_provider and env_provider.lower() in ["openai", "claude", "ollama"]:
            return env_provider.lower()  # type: ignore
        
        # Fall back to config file
        config_provider = self.config.get("llm", {}).get("default_provider", "openai")
        if config_provider in ["openai", "claude", "ollama"]:
            return config_provider  # type: ignore
        
        # Default to OpenAI if not specified
        return "openai"
    
    def get_provider_config(self, provider: Optional[ProviderType] = None) -> Dict[str, Any]:
        """Get configuration for a specific provider"""
        provider = provider or self.default_provider
        
        # Get provider config from the config file
        provider_config = self.config.get("llm", {}).get(provider, {})
        
        # Add API keys from environment variables
        if provider == "openai":
            provider_config["api_key"] = os.getenv("OPENAI_API_KEY", provider_config.get("api_key", ""))
        elif provider == "claude":
            provider_config["api_key"] = os.getenv("CLAUDE_API_KEY", provider_config.get("api_key", ""))
        
        return provider_config


class LLMProvider:
    """Unified interface for different LLM providers"""
    
    def __init__(self, provider: Optional[ProviderType] = None, config: Optional[Dict[str, Any]] = None):
        self.config_manager = LLMProviderConfig()
        self.provider = provider or self.config_manager.default_provider
        self.config = config or self.config_manager.get_provider_config(self.provider)
        
    def get_llm_config(self) -> Dict[str, Any]:
        """Get LLM configuration for crawl4ai LLMConfig"""
        if self.provider == "openai":
            return self._get_openai_config()
        elif self.provider == "claude":
            return self._get_claude_config()
        elif self.provider == "ollama":
            return self._get_ollama_config()
        else:
            # Default to OpenAI if provider is not recognized
            return self._get_openai_config()
    
    def _get_openai_config(self) -> Dict[str, Any]:
        """Get OpenAI configuration"""
        model = self.config.get("model", "gpt-3.5-turbo")
        temperature = self.config.get("temperature", 0.1)
        api_key = self.config.get("api_key", "")
        
        return {
            "provider": f"openai/{model}",
            "api_token": api_key,
            "extra_args": {"temperature": temperature}
        }
    
    def _get_claude_config(self) -> Dict[str, Any]:
        """Get Claude configuration"""
        model = self.config.get("model", "claude-3-sonnet-20240229")
        temperature = self.config.get("temperature", 0.0)
        api_key = self.config.get("api_key", "")
        base_url = self.config.get("base_url", "https://api.anthropic.com/v1/")
        
        return {
            "provider": f"anthropic/{model}",
            "api_token": api_key,
            "base_url": base_url,
            "extra_args": {"temperature": temperature}
        }
    
    def _get_ollama_config(self) -> Dict[str, Any]:
        """Get Ollama configuration"""
        model = self.config.get("model", "llama3.2:latest")
        temperature = self.config.get("temperature", 0.0)
        base_url = self.config.get("base_url", "http://localhost:11434/v1")
        
        return {
            "provider": f"ollama/{model}",
            "base_url": base_url,
            "extra_args": {"temperature": temperature}
        }
    
    def is_ollama_available(self) -> bool:
        """Check if Ollama is available and running"""
        if self.provider != "ollama":
            return False
        
        base_url = self.config.get("base_url", "http://localhost:11434")
        if not base_url.endswith("/v1"):
            api_url = f"{base_url.rstrip('/')}/api/tags"
        else:
            # Extract the base URL without /v1
            api_url = f"{base_url.rstrip('/').rsplit('/v1', 1)[0]}/api/tags"
        
        try:
            response = requests.get(api_url, timeout=2)
            return response.status_code == 200
        except Exception as e:
            logger.warning(f"Ollama not available: {e}")
            return False


# Helper function to get LLM configuration for crawl4ai
def get_llm_config(provider: Optional[ProviderType] = None) -> Dict[str, Any]:
    """Get LLM configuration for crawl4ai LLMConfig"""
    llm_provider = LLMProvider(provider)
    return llm_provider.get_llm_config()
