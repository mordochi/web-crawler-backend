import sqlite3
import os

def recreate_cache_table(db_path="protocol_cache.db"):
    """Recreate the protocol_cache table with the updated schema."""
    if os.path.exists(db_path):
        # Backup existing data if needed
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # Check if the old table exists
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='protocol_cache'")
        if cursor.fetchone():
            print("Backing up existing data...")
            try:
                # Get existing data
                cursor.execute("SELECT url, blockchain_id, data, timestamp, source FROM protocol_cache")
                existing_data = cursor.fetchall()
                print(f"Found {len(existing_data)} existing records")
                
                # Drop the existing table
                cursor.execute("DROP TABLE IF EXISTS protocol_cache")
                print("Dropped existing table")
                
                # Create the new table with updated schema
                cursor.execute("""
                CREATE TABLE protocol_cache (
                    url TEXT,
                    blockchain_id TEXT,
                    asset_symbols TEXT,
                    data TEXT,
                    timestamp REAL,
                    source TEXT,
                    PRIMARY KEY (url, blockchain_id, asset_symbols)
                )
                """)
                print("Created new table with updated schema")
                
                # Restore the data with NULL for asset_symbols
                if existing_data:
                    for record in existing_data:
                        url, blockchain_id, data, timestamp, source = record
                        cursor.execute(
                            "INSERT INTO protocol_cache (url, blockchain_id, asset_symbols, data, timestamp, source) VALUES (?, ?, NULL, ?, ?, ?)",
                            (url, blockchain_id, data, timestamp, source)
                        )
                    print(f"Restored {len(existing_data)} records with NULL asset_symbols")
            except Exception as e:
                print(f"Error during migration: {str(e)}")
        else:
            # Create the new table with updated schema
            cursor.execute("""
            CREATE TABLE protocol_cache (
                url TEXT,
                blockchain_id TEXT,
                asset_symbols TEXT,
                data TEXT,
                timestamp REAL,
                source TEXT,
                PRIMARY KEY (url, blockchain_id, asset_symbols)
            )
            """)
            print("Created new table with updated schema")
        
        conn.commit()
        conn.close()
        print("Database migration completed")
    else:
        print(f"Cache file {db_path} does not exist, creating new database")
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # Create the table with the new schema
        cursor.execute("""
        CREATE TABLE protocol_cache (
            url TEXT,
            blockchain_id TEXT,
            asset_symbols TEXT,
            data TEXT,
            timestamp REAL,
            source TEXT,
            PRIMARY KEY (url, blockchain_id, asset_symbols)
        )
        """)
        
        conn.commit()
        conn.close()
        print("Created new database with updated schema")

if __name__ == "__main__":
    recreate_cache_table()
