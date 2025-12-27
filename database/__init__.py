"""
Database package for Email Trust & Safety
"""

from .connection import get_db_manager, init_database
from .repository import get_repository
from .models import Email, DomainReputation, RiskFactor

__all__ = [
    'get_db_manager',
    'init_database', 
    'get_repository',
    'Email',
    'DomainReputation',
    'RiskFactor'
]