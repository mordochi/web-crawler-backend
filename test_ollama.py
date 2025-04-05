"""
Test script for Ollama integration
"""
import os
import sys
from llm_providers import LLMProvider, get_llm_config

def main():
    # Set environment variable for testing
    os.environ["LLM_PROVIDER"] = "ollama"
    
    # Create LLM provider
    provider = LLMProvider("ollama")
    
    # Check if Ollama is available
    is_available = provider.is_ollama_available()
    print(f"Ollama available: {is_available}")
    
    if not is_available:
        print("Ollama is not available. Please make sure it's running.")
        sys.exit(1)
    
    # Get LLM configuration
    llm_config = get_llm_config("ollama")
    print(f"LLM config: {llm_config}")
    
    # Test a simple request to Ollama
    import requests
    
    base_url = provider.config.get("base_url", "http://localhost:11434")
    if base_url.endswith("/v1"):
        base_url = base_url.rstrip("/v1")
    
    # Get list of models
    try:
        response = requests.get(f"{base_url}/api/tags")
        print(f"Models: {response.json()}")
    except Exception as e:
        print(f"Error getting models: {e}")
    
    # Test simple completion
    try:
        model = provider.config.get("model", "llama3.2:latest")
        response = requests.post(
            f"{base_url}/api/generate",
            json={
                "model": model,
                "prompt": "Extract key investment opportunities from the following text: 'Ethereum has several DeFi protocols including Aave, Uniswap, and Compound.'",
                "stream": False
            }
        )
        print(f"Completion response status: {response.status_code}")
        print(f"Completion: {response.json()}")
    except Exception as e:
        print(f"Error with completion: {e}")

if __name__ == "__main__":
    main()
