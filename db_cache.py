import sqlite3
import json
import os
import time
from datetime import datetime, timedelta

class DBCache:
    def __init__(self, db_path="cache.db", cache_expiry_hours=24):
        """Initialize the database cache.
        
        Args:
            db_path: Path to the SQLite database file
            cache_expiry_hours: Number of hours before cache entries expire
        """
        self.db_path = db_path
        self.cache_expiry_hours = cache_expiry_hours
        self._init_db()
    
    def _init_db(self):
        """Initialize the database with required tables if they don't exist."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Check if asset_symbols column exists
        cursor.execute("PRAGMA table_info(protocol_cache)")
        columns = [column[1] for column in cursor.fetchall()]
        
        if 'protocol_cache' not in columns:
            # Create cache table if it doesn't exist
            cursor.execute('''
            CREATE TABLE IF NOT EXISTS protocol_cache (
                url TEXT,
                blockchain_id TEXT,
                asset_symbols TEXT,
                data TEXT,
                timestamp REAL,
                source TEXT,
                PRIMARY KEY (url, blockchain_id, asset_symbols)
            )
            ''')
        elif 'asset_symbols' not in columns:
            # Add asset_symbols column if it doesn't exist
            cursor.execute("ALTER TABLE protocol_cache ADD COLUMN asset_symbols TEXT")
            # Update primary key to include asset_symbols
            cursor.execute("""CREATE TABLE protocol_cache_new (
                url TEXT,
                blockchain_id TEXT,
                asset_symbols TEXT,
                data TEXT,
                timestamp REAL,
                source TEXT,
                PRIMARY KEY (url, blockchain_id, asset_symbols)
            )""")
            cursor.execute("INSERT INTO protocol_cache_new SELECT url, blockchain_id, NULL, data, timestamp, source FROM protocol_cache")
            cursor.execute("DROP TABLE protocol_cache")
            cursor.execute("ALTER TABLE protocol_cache_new RENAME TO protocol_cache")
        
        conn.commit()
        conn.close()
    
    def get_cached_protocols(self, url, blockchain_id=None, asset_symbols=None):
        """Get cached protocols data for a URL and optional blockchain ID and asset symbols.
        
        Args:
            url: The URL that was crawled
            blockchain_id: Optional blockchain ID to filter by
            asset_symbols: Optional list of asset symbols to filter by
            
        Returns:
            The cached data as a dictionary/list or None if not found or expired
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Calculate expiry timestamp
        expiry_time = time.time() - (self.cache_expiry_hours * 3600)
        
        # Convert asset_symbols to a sorted, comma-separated string if provided
        asset_symbols_str = None
        if asset_symbols:
            if isinstance(asset_symbols, list):
                asset_symbols_str = ",".join(sorted(asset_symbols))
            else:
                asset_symbols_str = str(asset_symbols)
        
        if blockchain_id and asset_symbols_str:
            cursor.execute(
                "SELECT data, timestamp FROM protocol_cache WHERE url = ? AND blockchain_id = ? AND asset_symbols = ? AND timestamp > ?",
                (url, blockchain_id, asset_symbols_str, expiry_time)
            )
        elif blockchain_id:
            cursor.execute(
                "SELECT data, timestamp FROM protocol_cache WHERE url = ? AND blockchain_id = ? AND timestamp > ?",
                (url, blockchain_id, expiry_time)
            )
        elif asset_symbols_str:
            cursor.execute(
                "SELECT data, timestamp FROM protocol_cache WHERE url = ? AND asset_symbols = ? AND timestamp > ?",
                (url, asset_symbols_str, expiry_time)
            )
        else:
            cursor.execute(
                "SELECT data, timestamp FROM protocol_cache WHERE url = ? AND timestamp > ?",
                (url, expiry_time)
            )
        
        result = cursor.fetchone()
        conn.close()
        
        if result:
            data, timestamp = result
            cache_age_hours = (time.time() - timestamp) / 3600
            print(f"Using cached data from {cache_age_hours:.1f} hours ago")
            return json.loads(data)
        
        return None
    
    def cache_protocols(self, url, data, blockchain_id=None, asset_symbols=None, source="api"):
        """Cache protocols data for a URL and optional blockchain ID and asset symbols.
        
        Args:
            url: The URL that was crawled
            data: The data to cache (will be JSON serialized)
            blockchain_id: Optional blockchain ID to associate with the cache
            asset_symbols: Optional list of asset symbols to associate with the cache
            source: Source of the data (api, html, llm)
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Convert data to JSON string
        json_data = json.dumps(data)
        timestamp = time.time()
        
        # Convert asset_symbols to a sorted, comma-separated string if provided
        asset_symbols_str = None
        if asset_symbols:
            if isinstance(asset_symbols, list):
                asset_symbols_str = ",".join(sorted(asset_symbols))
            else:
                asset_symbols_str = str(asset_symbols)
        
        # Insert or replace cache entry
        cursor.execute(
            "INSERT OR REPLACE INTO protocol_cache (url, blockchain_id, asset_symbols, data, timestamp, source) VALUES (?, ?, ?, ?, ?, ?)",
            (url, blockchain_id, asset_symbols_str, json_data, timestamp, source)
        )
        
        conn.commit()
        conn.close()
        
        print(f"Cached protocols data for {url} ({blockchain_id}) with assets: {asset_symbols_str}")
    
    def clear_expired_cache(self):
        """Clear expired cache entries."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Calculate expiry timestamp
        expiry_time = time.time() - (self.cache_expiry_hours * 3600)
        
        cursor.execute("DELETE FROM protocol_cache WHERE timestamp < ?", (expiry_time,))
        deleted_count = cursor.rowcount
        
        conn.commit()
        conn.close()
        
        print(f"Cleared {deleted_count} expired cache entries")
        
    def get_cache_stats(self):
        """Get statistics about the cache."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("SELECT COUNT(*) FROM protocol_cache")
        total_entries = cursor.fetchone()[0]
        
        cursor.execute("SELECT blockchain_id, COUNT(*) FROM protocol_cache GROUP BY blockchain_id")
        blockchain_counts = cursor.fetchall()
        
        cursor.execute("SELECT source, COUNT(*) FROM protocol_cache GROUP BY source")
        source_counts = cursor.fetchall()
        
        conn.close()
        
        return {
            "total_entries": total_entries,
            "blockchain_counts": dict(blockchain_counts),
            "source_counts": dict(source_counts)
        }
