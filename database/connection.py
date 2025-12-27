"""
Database Connection Manager for Email Trust & Safety
Handles PostgreSQL connections and session management
"""

import os
import logging
from contextlib import contextmanager
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, scoped_session
from sqlalchemy.pool import QueuePool

from .models import Base

logger = logging.getLogger(__name__)


class DatabaseManager:
    """Manages database connections and sessions"""
    
    def __init__(self):
        self.engine = None
        self.Session = None
        self._initialized = False
    
    def initialize(self):
        """Initialize database connection"""
        if self._initialized:
            logger.info("Database already initialized")
            return
        
        # Get database configuration from environment
        db_user = os.getenv('DB_USER', 'postgres')
        db_password = os.getenv('DB_PASSWORD', 'postgres')
        db_host = os.getenv('DB_HOST', 'localhost')
        db_port = os.getenv('DB_PORT', '5432')
        db_name = os.getenv('DB_NAME', 'email_trust_safety')
        
        # Construct PostgreSQL connection URL
        database_url = f"postgresql://{db_user}:{db_password}@{db_host}:{db_port}/{db_name}"
        
        logger.info(f"Connecting to PostgreSQL: {db_host}:{db_port}/{db_name}")
        
        try:
            # Create engine with connection pooling
            self.engine = create_engine(
                database_url,
                poolclass=QueuePool,
                pool_size=10,
                max_overflow=20,
                pool_pre_ping=True,  # Verify connections before using
                echo=False  # Set to True for SQL query logging
            )
            
            # Create session factory
            session_factory = sessionmaker(bind=self.engine)
            self.Session = scoped_session(session_factory)
            
            self._initialized = True
            logger.info("✅ Database connection initialized successfully")
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize database: {e}")
            raise
    
    def create_tables(self):
        """Create all tables defined in models"""
        if not self._initialized:
            self.initialize()
        
        try:
            Base.metadata.create_all(self.engine)
            logger.info("✅ Database tables created/verified")
        except Exception as e:
            logger.error(f"❌ Failed to create tables: {e}")
            raise
    
    @contextmanager
    def get_session(self):
        """
        Context manager for database sessions
        
        Usage:
            with db.get_session() as session:
                # use session
                session.add(email)
                # auto-commits on exit
        """
        if not self._initialized:
            self.initialize()
        
        session = self.Session()
        try:
            yield session
            session.commit()
        except Exception as e:
            session.rollback()
            logger.error(f"Database session error: {e}")
            raise
        finally:
            session.close()
    
    def close(self):
        """Close database connections"""
        if self.Session:
            self.Session.remove()
        if self.engine:
            self.engine.dispose()
        self._initialized = False
        logger.info("Database connections closed")


# Singleton instance
_db_manager = None


def get_db_manager() -> DatabaseManager:
    """Get database manager singleton"""
    global _db_manager
    if _db_manager is None:
        _db_manager = DatabaseManager()
    return _db_manager


def init_database():
    """Initialize database and create tables"""
    db = get_db_manager()
    db.initialize()
    db.create_tables()
    logger.info("✅ Database initialization complete")