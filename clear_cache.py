import sqlite3
import os

def clear_all_cache(db_path="protocol_cache.db"):
    """Clear all cache entries."""
    if os.path.exists(db_path):
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # Get count before deletion
        cursor.execute("SELECT COUNT(*) FROM protocol_cache")
        count_before = cursor.fetchone()[0]
        
        # Delete all entries
        cursor.execute("DELETE FROM protocol_cache")
        
        conn.commit()
        conn.close()
        
        print(f"Cleared {count_before} cache entries")
    else:
        print(f"Cache file {db_path} does not exist")

if __name__ == "__main__":
    clear_all_cache()
