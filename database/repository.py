"""
Database Repository - Data Access Layer
Handles all database operations for email classifications
"""

from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta
from sqlalchemy import func, desc, and_, or_
from sqlalchemy.orm import Session
import logging

from .models import (
    Email, RiskFactor, DomainReputation, 
    SenderHistory, AbuseCampaign, AuditLog, PerformanceMetrics
)
from .connection import get_db_manager

logger = logging.getLogger(__name__)


class EmailRepository:
    """Repository for email classification operations"""
    
    def __init__(self):
        self.db = get_db_manager()
    
    def store_classification(self, classification_result: Dict[str, Any]) -> Email:
        """
        Store email classification result in database
        
        Args:
            classification_result: Dict from predictor.predict_email()
        
        Returns:
            Email: Created Email record
        """
        with self.db.get_session() as session:
            # Create email record
            email = Email(
                message_id=classification_result.get('message_id', ''),
                from_email=classification_result.get('from_email', ''),
                to_emails=classification_result.get('to', []),
                subject=classification_result.get('subject', ''),
                
                # Classification
                predicted_label=classification_result['predicted']['predicted_label'],
                ham_score=classification_result['predicted']['ham_score'],
                spam_score=classification_result['predicted']['spam_score'],
                phishing_score=classification_result['predicted']['phishing_score'],
                model_confidence=classification_result['predicted']['model_confidence'],
                model_version=classification_result['predicted'].get('model_used', 'xgboost_v3_enhanced'),
                
                # Risk analysis
                risk_score=classification_result['risk_analysis']['risk_score'],
                severity=classification_result['risk_analysis']['severity'],
                domain_risk_level=classification_result['risk_analysis']['domain_reputation']['risk_level'],
            )
            
            session.add(email)
            session.flush()  # Get email.id
            
            # Store risk factors
            for factor in classification_result.get('reasons', []):
                risk_factor = RiskFactor(
                    email_id=email.id,
                    factor_type=factor['type'],
                    weight=factor['weight'],
                    value=factor.get('value'),
                    description=factor['description']
                )
                session.add(risk_factor)
            
            # Update domain reputation
            self._update_domain_reputation(
                session,
                classification_result['from_email'],
                classification_result['predicted']['predicted_label'],
                classification_result['risk_analysis']
            )
            
            # Add to sender history
            self._add_sender_history(
                session,
                classification_result['from_email'],
                classification_result['message_id'],
                classification_result['predicted']['predicted_label'],
                classification_result['risk_analysis']['risk_score'],
                classification_result['predicted']['model_confidence']
            )
            
            # Detect campaigns
            self._detect_campaigns(session, email)
            
            logger.info(f"Stored classification: {email.message_id} - {email.predicted_label}")
            
            return email
    
    def _update_domain_reputation(
        self, 
        session: Session, 
        from_email: str, 
        label: str, 
        risk_analysis: Dict
    ):
        """Update domain reputation statistics"""
        if not from_email or '@' not in from_email:
            return
        
        domain = from_email.split('@')[1].lower()
        
        # Get or create domain reputation
        domain_rep = session.query(DomainReputation).filter_by(domain=domain).first()
        
        if not domain_rep:
            domain_rep = DomainReputation(
                domain=domain,
                risk_level=risk_analysis['domain_reputation']['risk_level'],
                risk_score=risk_analysis['domain_reputation']['risk_score'],
                is_free_provider=risk_analysis['domain_reputation']['is_free_provider'],
                risk_reasons=risk_analysis['domain_reputation']['reasons']
            )
            session.add(domain_rep)
        
        # Update counts
        domain_rep.total_emails += 1
        if label == 'ham':
            domain_rep.ham_count += 1
        elif label == 'spam':
            domain_rep.spam_count += 1
        elif label == 'phishing':
            domain_rep.phishing_count += 1
        
        domain_rep.last_seen = datetime.utcnow()
        
        # Recalculate risk level
        total = domain_rep.total_emails
        phish_ratio = domain_rep.phishing_count / total
        spam_ratio = domain_rep.spam_count / total
        
        if phish_ratio > 0.3 or spam_ratio > 0.5:
            domain_rep.risk_level = 'critical'
        elif phish_ratio > 0.1 or spam_ratio > 0.3:
            domain_rep.risk_level = 'high'
        elif phish_ratio > 0.05 or spam_ratio > 0.15:
            domain_rep.risk_level = 'medium'
        else:
            domain_rep.risk_level = 'low'
    
    def _add_sender_history(
        self, 
        session: Session, 
        email_address: str, 
        message_id: str, 
        label: str, 
        risk_score: int, 
        confidence: float
    ):
        """Add entry to sender history"""
        history = SenderHistory(
            email_address=email_address,
            message_id=message_id,
            predicted_label=label,
            risk_score=risk_score,
            confidence=confidence
        )
        session.add(history)
    
    def _detect_campaigns(self, session: Session, email: Email):
        """Detect coordinated abuse campaigns"""
        # Check for multiple emails from same domain in short time
        domain = email.from_email.split('@')[1] if '@' in email.from_email else None
        if not domain:
            return
        
        # Look for similar emails in last 10 minutes
        ten_min_ago = datetime.utcnow() - timedelta(minutes=10)
        
        similar_emails = session.query(Email).filter(
            Email.from_email.like(f'%@{domain}'),
            Email.classified_at >= ten_min_ago,
            Email.predicted_label.in_(['spam', 'phishing'])
        ).count()
        
        # If 15+ similar emails, flag as campaign
        if similar_emails >= 15:
            campaign = session.query(AbuseCampaign).filter_by(
                domain=domain,
                status='active'
            ).first()
            
            if not campaign:
                campaign = AbuseCampaign(
                    campaign_name=f"Campaign-{domain}-{datetime.utcnow().strftime('%Y%m%d')}",
                    domain=domain,
                    attack_type='coordinated_attack',
                    status='active',
                    severity='critical'
                )
                session.add(campaign)
            
            campaign.email_count = similar_emails
            campaign.last_detected = datetime.utcnow()
    
    def get_classification_stats(self, hours: int = 24) -> Dict:
        """Get classification statistics for time period"""
        with self.db.get_session() as session:
            start_time = datetime.utcnow() - timedelta(hours=hours)
            
            stats = session.query(
                Email.predicted_label,
                func.count(Email.id).label('count')
            ).filter(
                Email.classified_at >= start_time
            ).group_by(Email.predicted_label).all()
            
            return {label: count for label, count in stats}
    
    def get_top_malicious_domains(self, limit: int = 10) -> List[DomainReputation]:
        """Get top malicious domains"""
        with self.db.get_session() as session:
            return session.query(DomainReputation).filter(
                DomainReputation.risk_level.in_(['critical', 'high'])
            ).order_by(
                desc(DomainReputation.phishing_count + DomainReputation.spam_count)
            ).limit(limit).all()
    
    def get_recent_threats(self, limit: int = 20) -> List[Email]:
        """Get recent high-risk emails"""
        with self.db.get_session() as session:
            return session.query(Email).filter(
                Email.risk_score >= 60
            ).order_by(
                desc(Email.classified_at)
            ).limit(limit).all()
    
    def get_active_campaigns(self) -> List[AbuseCampaign]:
        """Get active abuse campaigns"""
        with self.db.get_session() as session:
            return session.query(AbuseCampaign).filter_by(
                status='active'
            ).order_by(desc(AbuseCampaign.last_detected)).all()
    
    def get_sender_reputation(self, email_address: str) -> Dict:
        """Get sender's historical behavior"""
        with self.db.get_session() as session:
            history = session.query(
                SenderHistory.predicted_label,
                func.count(SenderHistory.id).label('count'),
                func.avg(SenderHistory.risk_score).label('avg_risk')
            ).filter(
                SenderHistory.email_address == email_address
            ).group_by(SenderHistory.predicted_label).all()
            
            return {
                label: {'count': count, 'avg_risk': avg_risk}
                for label, count, avg_risk in history
            }
    
    def get_hourly_volume(self, hours: int = 24) -> List[Dict]:
        """Get email volume by hour"""
        with self.db.get_session() as session:
            start_time = datetime.utcnow() - timedelta(hours=hours)
            
            results = session.query(
                func.date_trunc('hour', Email.classified_at).label('hour'),
                Email.predicted_label,
                func.count(Email.id).label('count')
            ).filter(
                Email.classified_at >= start_time
            ).group_by(
                'hour', Email.predicted_label
            ).order_by('hour').all()
            
            return [
                {
                    'hour': r.hour,
                    'label': r.predicted_label,
                    'count': r.count
                }
                for r in results
            ]
    
    def search_emails(
        self, 
        from_email: str = None, 
        label: str = None, 
        min_risk: int = None,
        limit: int = 100
    ) -> List[Email]:
        """Search emails with filters"""
        with self.db.get_session() as session:
            query = session.query(Email)
            
            if from_email:
                query = query.filter(Email.from_email.like(f'%{from_email}%'))
            if label:
                query = query.filter(Email.predicted_label == label)
            if min_risk:
                query = query.filter(Email.risk_score >= min_risk)
            
            return query.order_by(desc(Email.classified_at)).limit(limit).all()


# Singleton
_repository = None


def get_repository() -> EmailRepository:
    """Get email repository instance"""
    global _repository
    if _repository is None:
        _repository = EmailRepository()
    return _repository