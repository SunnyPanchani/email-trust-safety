"""
FastAPI Response Models for Email Classification API
Ensures all fields are properly validated and returned
"""

from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional


class PredictedScores(BaseModel):
    ham_score: float = Field(..., description="Probability of being legitimate email")
    spam_score: float = Field(..., description="Probability of being spam")
    phishing_score: float = Field(..., description="Probability of being phishing")
    predicted_label: str = Field(..., description="Final classification: ham, spam, or phishing")
    model_used: str = Field(..., description="Model version used for prediction")
    model_confidence: float = Field(..., description="Confidence score of the prediction")


class RiskFactor(BaseModel):
    type: str = Field(..., description="Type of risk factor")
    weight: str = Field(..., description="Severity weight: critical, high, medium, low")
    value: Optional[Any] = Field(None, description="Additional value/data for this factor")
    description: str = Field(..., description="Human-readable description")


class DomainReputation(BaseModel):
    risk_level: str = Field(..., description="Domain risk level: critical, high, medium, low")
    risk_score: int = Field(..., description="Numerical risk score for domain")
    reasons: List[str] = Field(default_factory=list, description="List of risk reasons")
    is_free_provider: bool = Field(False, description="Whether it's a free email provider")
    domain: str = Field(..., description="The analyzed domain")


class RiskAnalysis(BaseModel):
    risk_score: int = Field(..., description="Overall risk score (0-100+)")
    severity: str = Field(..., description="Risk severity: critical, high, medium, low")
    domain_reputation: DomainReputation = Field(..., description="Domain reputation analysis")


class EmailClassificationResponse(BaseModel):
    message_id: str = Field(..., description="Unique message identifier")
    from_email: str = Field(..., description="Sender email address")
    to: List[str] = Field(..., description="Recipient email addresses")
    subject: str = Field(..., description="Email subject line")
    
    predicted: PredictedScores = Field(..., description="Prediction scores and classification")
    
    risk_analysis: RiskAnalysis = Field(..., description="Comprehensive risk analysis")
    
    metadata: Dict[str, Any] = Field(..., description="Email metadata features")
    
    reputation: Optional[Dict[str, Any]] = Field(None, description="Sender reputation data")
    
    reasons: List[RiskFactor] = Field(default_factory=list, description="List of risk factors detected")
    
    explanation: Optional[str] = Field(None, description="Human-readable explanation of classification")

    class Config:
        json_schema_extra = {
            "example": {
                "message_id": "abc123",
                "from_email": "suspicious@example.xyz",
                "to": ["victim@example.com"],
                "subject": "URGENT: Claim your prize NOW!",
                "predicted": {
                    "ham_score": 0.05,
                    "spam_score": 0.15,
                    "phishing_score": 0.80,
                    "predicted_label": "phishing",
                    "model_used": "xgboost_v3_enhanced",
                    "model_confidence": 0.95
                },
                "risk_analysis": {
                    "risk_score": 85,
                    "severity": "critical",
                    "domain_reputation": {
                        "risk_level": "high",
                        "risk_score": 3,
                        "reasons": ["high_risk_tld"],
                        "is_free_provider": False,
                        "domain": "example.xyz"
                    }
                },
                "metadata": {
                    "spam_keyword_count": 15,
                    "suspicious_sender": 1
                },
                "reputation": {
                    "sender_is_known": 0
                },
                "reasons": [
                    {
                        "type": "excessive_spam_keywords",
                        "weight": "critical",
                        "value": 15,
                        "description": "15 spam keywords detected"
                    }
                ],
                "explanation": "🚨 CRITICAL THREAT DETECTED - Multiple phishing indicators"
            }
        }