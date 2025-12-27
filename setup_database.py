#!/usr/bin/env python3
"""
Setup PostgreSQL Database for Email Trust & Safety
Creates database, tables, and indexes
"""

import os
import sys
import psycopg2
from psycopg2.extensions import ISOLATION_LEVEL_AUTOCOMMIT
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
from dotenv import load_dotenv
load_dotenv()



def create_database():
    """Create the PostgreSQL database if it doesn't exist"""
    
    # Connection parameters
    db_user = os.getenv('DB_USER', 'postgres')
    db_password = os.getenv('DB_PASSWORD', 'postgres')
    db_host = os.getenv('DB_HOST', 'localhost')
    db_port = os.getenv('DB_PORT', '5432')
    db_name = os.getenv('DB_NAME', 'email_trust_safety')
    
    logger.info(f"Connecting to PostgreSQL at {db_host}:{db_port}")
    
    try:
        # Connect to default 'postgres' database
        conn = psycopg2.connect(
            dbname='postgres',
            user=db_user,
            password=db_password,
            host=db_host,
            port=db_port
        )
        conn.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)
        cursor = conn.cursor()
        
        # Check if database exists
        cursor.execute(
            "SELECT 1 FROM pg_database WHERE datname = %s",
            (db_name,)
        )
        
        if cursor.fetchone():
            logger.info(f"Database '{db_name}' already exists")
        else:
            # Create database
            cursor.execute(f'CREATE DATABASE {db_name}')
            logger.info(f"✓ Created database '{db_name}'")
        
        cursor.close()
        conn.close()
        
        return True
        
    except psycopg2.Error as e:
        logger.error(f"✗ Database creation failed: {e}")
        return False


def create_tables():
    """Create all tables using SQLAlchemy"""
    try:
        # Import after database is created
        from database.connection import init_database
        
        logger.info("Creating database tables...")
        init_database()
        logger.info("✓ All tables created successfully")
        
        return True
        
    except Exception as e:
        logger.error(f"✗ Table creation failed: {e}")
        return False


def verify_setup():
    """Verify database setup"""
    try:
        from database.connection import get_db_manager
        from database.models import Email, DomainReputation
        
        db = get_db_manager()
        
        with db.get_session() as session:
            # Try to count emails
            count = session.query(Email).count()
            logger.info(f"✓ Database connection verified ({count} emails in database)")
            
            # Try to count domains
            domain_count = session.query(DomainReputation).count()
            logger.info(f"✓ Domain reputation table verified ({domain_count} domains tracked)")
        
        return True
        
    except Exception as e:
        logger.error(f"✗ Verification failed: {e}")
        return False


def main():
    """Main setup function"""
    logger.info("=" * 60)
    logger.info("📊 PostgreSQL Database Setup")
    logger.info("=" * 60)
    
    # Check environment variables
    db_name = os.getenv('DB_NAME', 'email_trust_safety')
    db_user = os.getenv('DB_USER', 'postgres')
    db_host = os.getenv('DB_HOST', 'localhost')
    
    logger.info(f"Database: {db_name}")
    logger.info(f"User: {db_user}")
    logger.info(f"Host: {db_host}")
    logger.info("")
    
    # Step 1: Create database
    logger.info("Step 1: Creating database...")
    if not create_database():
        logger.error("Failed to create database. Exiting.")
        sys.exit(1)
    
    logger.info("")
    
    # Step 2: Create tables
    logger.info("Step 2: Creating tables...")
    if not create_tables():
        logger.error("Failed to create tables. Exiting.")
        sys.exit(1)
    
    logger.info("")
    
    # Step 3: Verify setup
    logger.info("Step 3: Verifying setup...")
    if not verify_setup():
        logger.error("Verification failed. Please check the logs.")
        sys.exit(1)
    
    logger.info("")
    logger.info("=" * 60)
    logger.info("✓ Database setup completed successfully!")
    logger.info("=" * 60)
    logger.info("")
    logger.info("Next steps:")
    logger.info("1. Start the API: cd api && python main.py")
    logger.info("2. Classifications will be automatically stored in PostgreSQL")
    logger.info("3. View dashboard for real-time metrics")
    logger.info("")


if __name__ == "__main__":
    main()