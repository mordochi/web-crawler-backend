from db_cache import DBCache
import json

def test_cache_with_asset_symbols():
    """Test caching with different asset symbols combinations."""
    cache = DBCache('protocol_cache.db')
    
    # Sample data
    test_url = "https://test-api.example.com/protocols"
    test_data = [
        {"name": "Protocol1", "tvl": 1000000, "chains": ["ethereum"]},
        {"name": "Protocol2", "tvl": 2000000, "chains": ["ethereum", "binance-smart-chain"]},
        {"name": "Protocol3", "tvl": 3000000, "chains": ["ethereum", "polygon"]}
    ]
    
    # Test case 1: Cache with blockchain_id only
    print("\n--- Test 1: Cache with blockchain_id only ---")
    cache.cache_protocols(test_url, test_data, blockchain_id="ethereum", source="test")
    
    # Verify cache
    cached_data = cache.get_cached_protocols(test_url, blockchain_id="ethereum")
    print(f"Retrieved data with blockchain_id only: {cached_data is not None}")
    
    # Test case 2: Cache with blockchain_id and asset_symbols
    print("\n--- Test 2: Cache with blockchain_id and asset_symbols ---")
    asset_symbols = ["ETH", "USDC", "DAI"]
    cache.cache_protocols(test_url, test_data, blockchain_id="ethereum", asset_symbols=asset_symbols, source="test")
    
    # Verify cache with same asset symbols
    cached_data = cache.get_cached_protocols(test_url, blockchain_id="ethereum", asset_symbols=asset_symbols)
    print(f"Retrieved data with matching asset_symbols: {cached_data is not None}")
    
    # Test case 3: Try to retrieve with different asset symbols
    print("\n--- Test 3: Retrieve with different asset_symbols ---")
    different_symbols = ["ETH", "USDT", "WBTC"]
    cached_data = cache.get_cached_protocols(test_url, blockchain_id="ethereum", asset_symbols=different_symbols)
    print(f"Retrieved data with different asset_symbols: {cached_data is not None}")
    
    # Test case 4: Cache with different blockchain_id but same asset symbols
    print("\n--- Test 4: Cache with different blockchain_id but same asset_symbols ---")
    cache.cache_protocols(test_url, test_data, blockchain_id="polygon", asset_symbols=asset_symbols, source="test")
    
    # Verify cache with different blockchain_id
    cached_data = cache.get_cached_protocols(test_url, blockchain_id="polygon", asset_symbols=asset_symbols)
    print(f"Retrieved data with different blockchain_id: {cached_data is not None}")
    
    # Test case 5: Retrieve with asset_symbols only
    print("\n--- Test 5: Retrieve with asset_symbols only ---")
    cached_data = cache.get_cached_protocols(test_url, asset_symbols=asset_symbols)
    print(f"Retrieved data with asset_symbols only: {cached_data is not None}")
    
    # Print cache stats
    print("\n--- Cache Stats ---")
    stats = cache.get_cache_stats()
    print(json.dumps(stats, indent=2))

if __name__ == "__main__":
    test_cache_with_asset_symbols()
