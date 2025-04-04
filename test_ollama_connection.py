#!/usr/bin/env python3
"""
Test script to verify Ollama connection for the web crawler backend.
This script checks if Ollama is running and available.
"""

import os
import sys
import requests
import json
from dotenv import load_dotenv
from llm_providers import LLMProvider

# Load environment variables
load_dotenv()

def test_ollama_connection():
    """Test if Ollama is running and available"""
    print("Testing Ollama connection...")
    
    # Create LLM provider with Ollama
    provider = LLMProvider(provider="ollama")
    
    # Check if Ollama is available
    if provider.is_ollama_available():
        print("✅ Ollama is available!")
        
        # Get the base URL
        base_url = provider.config.get("base_url", "http://localhost:11434/v1")
        if not base_url.endswith("/v1"):
            api_url = f"{base_url.rstrip('/')}/api/tags"
        else:
            # Extract the base URL without /v1
            api_url = f"{base_url.rstrip('/').rsplit('/v1', 1)[0]}/api/tags"
        
        # Get available models
        try:
            response = requests.get(api_url, timeout=5)
            if response.status_code == 200:
                models = response.json().get("models", [])
                if models:
                    print(f"Available models: {len(models)}")
                    for model in models:
                        print(f"  - {model.get('name')} (Size: {model.get('size', 'unknown')})")
                else:
                    print("No models found. You may need to pull models using 'ollama pull <model_name>'")
                
                # Check if our configured model is available
                model_name = provider.config.get("model", "llama3.2:latest")
                model_base = model_name.split(":")[0]
                
                model_available = any(model.get("name").startswith(model_base) for model in models)
                if model_available:
                    print(f"✅ Configured model '{model_name}' (or similar) is available")
                else:
                    print(f"❌ Configured model '{model_name}' not found. Pull it with:")
                    print(f"   ollama pull {model_name}")
            else:
                print(f"❌ Failed to get models list: {response.status_code}")
        except Exception as e:
            print(f"❌ Error getting models: {e}")
    else:
        print("❌ Ollama is not available. Make sure Ollama is running with:")
        print("   ollama serve")
        return False
    
    # Test a simple completion to verify the API works
    try:
        print("\nTesting a simple completion...")
        model_name = provider.config.get("model", "llama3.2:latest")
        base_url = provider.config.get("base_url", "http://localhost:11434/v1")
        
        if not base_url.endswith("/v1"):
            completion_url = f"{base_url.rstrip('/')}/api/generate"
            data = {
                "model": model_name,
                "prompt": "Say hello in one short sentence.",
                "stream": False
            }
        else:
            completion_url = f"{base_url}/chat/completions"
            data = {
                "model": model_name,
                "messages": [{"role": "user", "content": "Say hello in one short sentence."}],
                "stream": False
            }
        
        response = requests.post(completion_url, json=data, timeout=10)
        
        if response.status_code == 200:
            if not base_url.endswith("/v1"):
                result = response.json().get("response", "")
                print(f"Response: {result}")
            else:
                result = response.json().get("choices", [{}])[0].get("message", {}).get("content", "")
                print(f"Response: {result}")
            print("✅ Ollama API is working correctly!")
            return True
        else:
            print(f"❌ API test failed with status code: {response.status_code}")
            print(f"Response: {response.text}")
            return False
    except Exception as e:
        print(f"❌ Error testing completion: {e}")
        return False

def main():
    """Main function"""
    print("Ollama Connection Test for Web Crawler Backend")
    print("=============================================")
    
    result = test_ollama_connection()
    
    print("\nSummary:")
    if result:
        print("✅ Ollama is properly configured and working!")
        print("You can use the web crawler with Ollama as the LLM provider.")
        print("Set LLM_PROVIDER=ollama in your .env file to use Ollama.")
    else:
        print("❌ Ollama connection test failed.")
        print("Please check that Ollama is installed and running.")
        print("You can install Ollama from: https://ollama.com/download")
        print("Run 'ollama serve' to start the Ollama server.")
        print("Then pull the required model with: ollama pull llama3.2:latest")
    
    return 0 if result else 1

if __name__ == "__main__":
    sys.exit(main())
