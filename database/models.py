"""
SQLAlchemy Models for Email Trust & Safety Database
PostgreSQL schema for storing classifications, tracking abuse, and investigations
"""

from sqlalchemy import (
    Column, Integer, String, Float, Text, DateTime, Boolean,
    ForeignKey, Index, JSON, func
)
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
from datetime import datetime

Base = declarative_base()


class Email(Base):
    """Main table for email classifications"""
    __tablename__ = 'emails'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    message_id = Column(String(255), unique=True, nullable=False, index=True)
    
    # Email metadata
    from_email = Column(String(255), nullable=False, index=True)
    to_emails = Column(JSON)  # Store as array
    cc_emails = Column(JSON)
    bcc_emails = Column(JSON)
    subject = Column(Text)
    body = Column(Text)
    headers = Column(JSON)
    
    # Classification results
    predicted_label = Column(String(20), nullable=False, index=True)  # ham, spam, phishing
    ham_score = Column(Float)
    spam_score = Column(Float)
    phishing_score = Column(Float)
    model_confidence = Column(Float)
    model_version = Column(String(50), default='xgboost_v3_enhanced')
    
    # Risk analysis
    risk_score = Column(Integer, index=True)
    severity = Column(String(20), index=True)  # critical, high, medium, low
    domain_risk_level = Column(String(20))
    
    # Timestamps
    classified_at = Column(DateTime, default=datetime.utcnow, nullable=False, index=True)
    email_sent_at = Column(DateTime)
    
    # Relationships
    risk_factors = relationship("RiskFactor", back_populates="email", cascade="all, delete-orphan")
    
    # Indexes for common queries
    __table_args__ = (
        Index('idx_from_label_time', 'from_email', 'predicted_label', 'classified_at'),
        Index('idx_risk_severity', 'risk_score', 'severity'),
        Index('idx_classified_date', 'classified_at'),
    )
    
    def __repr__(self):
        return f"<Email(id={self.id}, from={self.from_email}, label={self.predicted_label})>"


class RiskFactor(Base):
    """Risk factors detected in emails"""
    __tablename__ = 'risk_factors'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    email_id = Column(Integer, ForeignKey('emails.id', ondelete='CASCADE'), nullable=False, index=True)
    
    factor_type = Column(String(50), nullable=False, index=True)
    weight = Column(String(20))  # critical, high, medium, low
    value = Column(JSON)  # Store complex data as JSON
    description = Column(Text)
    
    detected_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationship
    email = relationship("Email", back_populates="risk_factors")
    
    __table_args__ = (
        Index('idx_factor_type_weight', 'factor_type', 'weight'),
    )
    
    def __repr__(self):
        return f"<RiskFactor(type={self.factor_type}, weight={self.weight})>"


class DomainReputation(Base):
    """Track domain reputation over time"""
    __tablename__ = 'domain_reputation'
    
    domain = Column(String(255), primary_key=True)
    
    # Email counts
    total_emails = Column(Integer, default=0)
    ham_count = Column(Integer, default=0)
    spam_count = Column(Integer, default=0)
    phishing_count = Column(Integer, default=0)
    
    # Risk metrics
    risk_level = Column(String(20), index=True)  # critical, high, medium, low
    risk_score = Column(Integer)
    avg_risk_score = Column(Float)
    
    # Domain analysis
    is_free_provider = Column(Boolean, default=False)
    has_suspicious_tld = Column(Boolean, default=False)
    risk_reasons = Column(JSON)  # Array of reasons
    
    # Timestamps
    first_seen = Column(DateTime, default=datetime.utcnow)
    last_seen = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    last_updated = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    __table_args__ = (
        Index('idx_risk_level_count', 'risk_level', 'total_emails'),
        Index('idx_last_seen', 'last_seen'),
    )
    
    def __repr__(self):
        return f"<DomainReputation(domain={self.domain}, risk={self.risk_level})>"


class SenderHistory(Base):
    """Track individual sender behavior"""
    __tablename__ = 'sender_history'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    
    email_address = Column(String(255), nullable=False, index=True)
    message_id = Column(String(255), ForeignKey('emails.message_id'))
    
    # Classification
    predicted_label = Column(String(20))
    risk_score = Column(Integer)
    confidence = Column(Float)
    
    # Timestamps
    sent_at = Column(DateTime, default=datetime.utcnow, index=True)
    
    __table_args__ = (
        Index('idx_sender_time', 'email_address', 'sent_at'),
        Index('idx_sender_label', 'email_address', 'predicted_label'),
    )
    
    def __repr__(self):
        return f"<SenderHistory(email={self.email_address}, label={self.predicted_label})>"


class AbuseCampaign(Base):
    """Track coordinated abuse campaigns"""
    __tablename__ = 'abuse_campaigns'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    
    campaign_name = Column(String(255))
    domain = Column(String(255), index=True)
    attack_type = Column(String(50), index=True)  # lottery_scam, account_phishing, etc.
    
    # Campaign metrics
    email_count = Column(Integer, default=0)
    unique_senders = Column(Integer, default=0)
    unique_targets = Column(Integer, default=0)
    
    # Pattern indicators
    common_subject_pattern = Column(Text)
    common_keywords = Column(JSON)  # Array of keywords
    avg_risk_score = Column(Float)
    
    # Timestamps
    first_detected = Column(DateTime, default=datetime.utcnow, index=True)
    last_detected = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Status tracking
    status = Column(String(20), default='active', index=True)  # active, contained, resolved
    severity = Column(String(20))  # critical, high, medium
    
    # Investigation
    notes = Column(Text)
    assigned_to = Column(String(100))
    resolved_at = Column(DateTime)
    
    __table_args__ = (
        Index('idx_campaign_status', 'status', 'severity'),
        Index('idx_campaign_domain', 'domain', 'attack_type'),
    )
    
    def __repr__(self):
        return f"<AbuseCampaign(name={self.campaign_name}, type={self.attack_type})>"


class AuditLog(Base):
    """Audit trail for compliance and investigation"""
    __tablename__ = 'audit_logs'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    
    # Event details
    event_type = Column(String(50), nullable=False, index=True)  # classification, investigation, block, etc.
    event_action = Column(String(100))
    
    # Related entities
    email_id = Column(Integer, ForeignKey('emails.id'))
    message_id = Column(String(255))
    user_email = Column(String(255))
    
    # Event data
    event_data = Column(JSON)  # Flexible data storage
    
    # Context
    ip_address = Column(String(45))
    user_agent = Column(Text)
    
    # Timestamp
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False, index=True)
    
    __table_args__ = (
        Index('idx_audit_type_time', 'event_type', 'created_at'),
        Index('idx_audit_email', 'email_id'),
    )
    
    def __repr__(self):
        return f"<AuditLog(type={self.event_type}, action={self.event_action})>"


class PerformanceMetrics(Base):
    """Track model performance over time"""
    __tablename__ = 'performance_metrics'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    
    # Time window
    metric_date = Column(DateTime, nullable=False, index=True)
    time_window = Column(String(20))  # hourly, daily, weekly
    
    # Volume metrics
    total_emails = Column(Integer)
    ham_count = Column(Integer)
    spam_count = Column(Integer)
    phishing_count = Column(Integer)
    
    # Performance metrics
    avg_confidence = Column(Float)
    avg_processing_time_ms = Column(Float)
    
    # Risk metrics
    high_risk_count = Column(Integer)
    critical_risk_count = Column(Integer)
    avg_risk_score = Column(Float)
    
    # False positive tracking (if feedback available)
    false_positives = Column(Integer, default=0)
    false_negatives = Column(Integer, default=0)
    
    created_at = Column(DateTime, default=datetime.utcnow)
    
    __table_args__ = (
        Index('idx_metrics_date', 'metric_date', 'time_window'),
    )
    
    def __repr__(self):
        return f"<PerformanceMetrics(date={self.metric_date}, window={self.time_window})>"