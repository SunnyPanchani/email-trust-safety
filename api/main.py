

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import os
import logging
from predictor import get_predictor  # <-- enhanced predictor
from groq_explainer import GroqExplainer
from schemas import EmailRequest, BatchRequest, RiskReportResponse
from response_models import EmailClassificationResponse

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("api.main")

app = FastAPI(
    title="Email Trust & Safety API (v3 Enhanced)",
    description="Enterprise anti-abuse API with enhanced detection (XGBoost_v3 + Multi-factor Risk Analysis)",
    version="3.0.0-enhanced",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ----------------------------------------------------------
# Initialize Enhanced Predictor v3
# ----------------------------------------------------------
predictor = get_predictor()  # <-- Uses enhanced version with risk scoring
explainer = GroqExplainer(api_key=os.getenv("GROQ_API_KEY"))


@app.get("/")
def read_root():
    return {
        "service": "Email Trust & Safety API",
        "version": "3.0.0-enhanced",
        "status": "running",
        "model": "xgboost_v3_enhanced",
        "features": [
            "Multi-factor risk analysis",
            "Domain reputation scoring",
            "Enhanced phishing detection",
            "40+ spam keyword detection",
            "Aggressive auto-classification",
            "Groq AI explanations"
        ]
    }


@app.get("/health")
def health():
    return {
        "status": "ok",
        "model": "xgboost_v3_enhanced",
        "features_loaded": True,
        "groq_available": explainer.is_available()
    }


# ----------------------------------------------------------
# SCORE SINGLE EMAIL (Enhanced Response)
# ----------------------------------------------------------
@app.post("/score_email", response_model=EmailClassificationResponse)
def score_email(payload: EmailRequest):
    """
    Score a single email with enhanced risk analysis
    
    Returns:
    - Prediction scores (ham, spam, phishing)
    - Risk analysis with severity levels
    - Domain reputation scoring
    - Detailed risk factors
    - AI-generated explanation (if Groq available)
    """
    try:
        # Convert to dict for predictor
        email_dict = payload.dict()
        
        # Get prediction with enhanced risk analysis
        result = predictor.predict_email(email_dict)
        
        # Add Groq explanation if available
        if explainer.is_available():
            try:
                explanation = explainer.explain({
                    "from": payload.from_email or payload.dict().get("from", ""),
                    "subject": payload.subject,
                    "body": payload.body,
                    "model_prediction": result["predicted"],
                    "risk_analysis": result.get("risk_analysis", {}),
                    "reasons": result.get("reasons", [])
                })
                result["explanation"] = explanation
            except Exception as e:
                logger.warning(f"Groq explanation failed: {e}")
                # Keep the model's explanation if Groq fails
        
        # Ensure risk_analysis exists (defensive programming)
        if "risk_analysis" not in result:
            result["risk_analysis"] = {
                "risk_score": 0,
                "severity": "low",
                "domain_reputation": {
                    "risk_level": "low",
                    "risk_score": 0,
                    "reasons": [],
                    "is_free_provider": False,
                    "domain": result.get("from_email", "unknown")
                }
            }
        
        return result
        
    except Exception as e:
        logger.exception("score_email failed")
        raise HTTPException(status_code=500, detail=str(e))


# ----------------------------------------------------------
# SCORE BATCH
# ----------------------------------------------------------
@app.post("/score_batch")
def score_batch(payload: BatchRequest):
    """
    Score multiple emails in batch
    """
    try:
        results = []
        for email in payload.emails:
            try:
                r = predictor.predict_email(email.dict())
                
                # Optionally add Groq explanation for each
                if explainer.is_available():
                    try:
                        explanation = explainer.explain({
                            "from": email.from_email or email.dict().get("from", ""),
                            "subject": email.subject,
                            "body": email.body,
                            "model_prediction": r["predicted"],
                            "risk_analysis": r.get("risk_analysis", {}),
                            "reasons": r.get("reasons", [])
                        })
                        r["explanation"] = explanation
                    except:
                        pass  # Skip explanation on error
                
                results.append(r)
            except Exception as e:
                logger.error(f"Failed to process email: {e}")
                results.append({
                    "error": str(e),
                    "message_id": email.dict().get("message_id", "unknown")
                })
        
        return {"count": len(results), "results": results}
        
    except Exception as e:
        logger.exception("score_batch failed")
        raise HTTPException(status_code=500, detail=str(e))


# ----------------------------------------------------------
# ANALYZE LINKS ENDPOINT
# ----------------------------------------------------------
@app.post("/analyze_links")
def analyze_links(payload: EmailRequest):
    """
    Extract and analyze URLs in email
    """
    try:
        import re
        body = payload.body or ""
        
        # Extract URLs
        urls = re.findall(r"https?://\S+", body)
        
        # Analyze each URL
        url_analysis = []
        for url in urls:
            analysis = {
                "url": url,
                "is_https": url.startswith("https"),
                "has_ip": bool(re.search(r"https?://\d+\.\d+\.\d+\.\d+", url)),
                "suspicious_tld": any(url.endswith(tld) for tld in 
                                     [".xyz", ".top", ".ru", ".cn", ".tk", ".ml"]),
                "is_shortened": any(short in url for short in 
                                   ["bit.ly", "tinyurl", "t.co", "goo.gl"]),
                "has_hex_encoding": "%" in url,
                "length": len(url)
            }
            url_analysis.append(analysis)
        
        # Calculate overall risk
        high_risk_count = sum(1 for u in url_analysis 
                             if u["has_ip"] or u["suspicious_tld"] or u["is_shortened"])
        
        return {
            "url_count": len(urls),
            "urls": url_analysis,
            "high_risk_urls": high_risk_count,
            "risk_level": "high" if high_risk_count > 0 else "low"
        }
        
    except Exception as e:
        logger.exception("analyze_links failed")
        raise HTTPException(status_code=500, detail=str(e))


# ----------------------------------------------------------
# RISK REPORT (alias for score_email)
# ----------------------------------------------------------
@app.post("/risk_report", response_model=EmailClassificationResponse)
def risk_report(payload: EmailRequest):
    """
    Generate comprehensive risk report for email
    (Alias for score_email with same enhanced analysis)
    """
    return score_email(payload)


# ----------------------------------------------------------
# NEW: Get Risk Score Only
# ----------------------------------------------------------
@app.post("/risk_score")
def risk_score(payload: EmailRequest):
    """
    Get just the risk score and severity (lightweight endpoint)
    """
    try:
        result = predictor.predict_email(payload.dict())
        
        return {
            "message_id": result.get("message_id"),
            "predicted_label": result["predicted"]["predicted_label"],
            "risk_score": result.get("risk_analysis", {}).get("risk_score", 0),
            "severity": result.get("risk_analysis", {}).get("severity", "low"),
            "confidence": result["predicted"]["model_confidence"]
        }
        
    except Exception as e:
        logger.exception("risk_score failed")
        raise HTTPException(status_code=500, detail=str(e))


# ----------------------------------------------------------
# NEW: Domain Reputation Check
# ----------------------------------------------------------
@app.get("/domain_reputation/{domain}")
def check_domain_reputation(domain: str):
    """
    Check reputation of a specific domain
    """
    try:
        from predictor import analyze_domain_reputation
        
        # Create fake email to analyze domain
        fake_email = f"test@{domain}"
        analysis = analyze_domain_reputation(fake_email)
        
        return {
            "domain": domain,
            "reputation": analysis
        }
        
    except Exception as e:
        logger.exception("domain_reputation check failed")
        raise HTTPException(status_code=500, detail=str(e))