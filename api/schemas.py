




from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any

class EmailRequest(BaseModel):
    message_id: Optional[str] = None
    from_email: Optional[str] = Field(None, alias="from")
    to: Optional[List[str]] = None
    cc: Optional[List[str]] = None
    bcc: Optional[List[str]] = None
    subject: Optional[str] = ""
    body: Optional[str] = ""
    date: Optional[str] = None
    headers: Optional[Dict[str, str]] = None

    class Config:
        populate_by_name = True  # Allow both 'from' and 'from_email'


class BatchRequest(BaseModel):
    emails: List[EmailRequest]


class PredictedScore(BaseModel):
    """Prediction scores and label"""
    spam_score: float
    phishing_score: float
    ham_score: float
    predicted_label: str
    model_used: Optional[str] = None
    model_confidence: Optional[float] = None


class RiskReportResponse(BaseModel):
    """Complete risk assessment response"""
    message_id: Optional[str] = None
    from_email: Optional[str] = None
    to: Optional[List[str]] = None
    subject: Optional[str] = None
    predicted: PredictedScore
    metadata: Dict[str, Any]
    reasons: List[Dict[str, Any]]
    explanation: Optional[str] = None


# Keep old schemas for backward compatibility (if needed elsewhere)
class ScoreDict(BaseModel):
    spam_score: float
    phishing_score: float
    ham_score: float
    predicted_label: str
    probabilities: Dict[str, float]  # This was required but not provided