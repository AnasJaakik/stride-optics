#!/usr/bin/env python3
"""
Database migration script to add new columns to existing database.
Run this once to update your database schema.
"""
import sqlite3
import os
from pathlib import Path

# Get database path
BASE_DIR = Path(__file__).parent.parent
DB_PATH = BASE_DIR / 'gait_analysis.db'

if not DB_PATH.exists():
    print(f"Database not found at {DB_PATH}")
    print("The database will be created automatically on first run.")
    exit(0)

print(f"Migrating database at {DB_PATH}...")

conn = sqlite3.connect(str(DB_PATH))
cursor = conn.cursor()

# Check if columns exist and add them if they don't
try:
    # Get current columns
    cursor.execute("PRAGMA table_info(results)")
    columns = [row[1] for row in cursor.fetchall()]
    
    # Add missing columns
    if 'cadence_data' not in columns:
        print("Adding cadence_data column...")
        cursor.execute("ALTER TABLE results ADD COLUMN cadence_data TEXT")
    
    if 'ground_contact_time_data' not in columns:
        print("Adding ground_contact_time_data column...")
        cursor.execute("ALTER TABLE results ADD COLUMN ground_contact_time_data TEXT")
    
    if 'overlay_video_path' not in columns:
        print("Adding overlay_video_path column...")
        cursor.execute("ALTER TABLE results ADD COLUMN overlay_video_path VARCHAR(255)")
    
    conn.commit()
    print("✓ Migration completed successfully!")
    
except Exception as e:
    print(f"Error during migration: {e}")
    conn.rollback()
    raise
finally:
    conn.close()
