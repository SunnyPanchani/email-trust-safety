

# from fastapi import FastAPI, HTTPException
# from fastapi.middleware.cors import CORSMiddleware
# import os
# import logging
# from predictor import get_predictor  # <-- enhanced predictor
# from groq_explainer import GroqExplainer
# from schemas import EmailRequest, BatchRequest, RiskReportResponse
# from response_models import EmailClassificationResponse

# logging.basicConfig(level=logging.INFO)
# logger = logging.getLogger("api.main")

# app = FastAPI(
#     title="Email Trust & Safety API (v3 Enhanced)",
#     description="Enterprise anti-abuse API with enhanced detection (XGBoost_v3 + Multi-factor Risk Analysis)",
#     version="3.0.0-enhanced",
# )

# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["*"],
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )

# # ----------------------------------------------------------
# # Initialize Enhanced Predictor v3
# # ----------------------------------------------------------
# predictor = get_predictor()  # <-- Uses enhanced version with risk scoring
# explainer = GroqExplainer(api_key=os.getenv("GROQ_API_KEY"))


# @app.get("/")
# def read_root():
#     return {
#         "service": "Email Trust & Safety API",
#         "version": "3.0.0-enhanced",
#         "status": "running",
#         "model": "xgboost_v3_enhanced",
#         "features": [
#             "Multi-factor risk analysis",
#             "Domain reputation scoring",
#             "Enhanced phishing detection",
#             "40+ spam keyword detection",
#             "Aggressive auto-classification",
#             "Groq AI explanations"
#         ]
#     }


# @app.get("/health")
# def health():
#     return {
#         "status": "ok",
#         "model": "xgboost_v3_enhanced",
#         "features_loaded": True,
#         "groq_available": explainer.is_available()
#     }


# # ----------------------------------------------------------
# # SCORE SINGLE EMAIL (Enhanced Response)
# # ----------------------------------------------------------
# @app.post("/score_email", response_model=EmailClassificationResponse)
# def score_email(payload: EmailRequest):
#     """
#     Score a single email with enhanced risk analysis
    
#     Returns:
#     - Prediction scores (ham, spam, phishing)
#     - Risk analysis with severity levels
#     - Domain reputation scoring
#     - Detailed risk factors
#     - AI-generated explanation (if Groq available)
#     """
#     try:
#         # Convert to dict for predictor
#         email_dict = payload.dict()
        
#         # Get prediction with enhanced risk analysis
#         result = predictor.predict_email(email_dict)
        
#         # Add Groq explanation if available
#         if explainer.is_available():
#             try:
#                 explanation = explainer.explain({
#                     "from": payload.from_email or payload.dict().get("from", ""),
#                     "subject": payload.subject,
#                     "body": payload.body,
#                     "model_prediction": result["predicted"],
#                     "risk_analysis": result.get("risk_analysis", {}),
#                     "reasons": result.get("reasons", [])
#                 })
#                 result["explanation"] = explanation
#             except Exception as e:
#                 logger.warning(f"Groq explanation failed: {e}")
#                 # Keep the model's explanation if Groq fails
        
#         # Ensure risk_analysis exists (defensive programming)
#         if "risk_analysis" not in result:
#             result["risk_analysis"] = {
#                 "risk_score": 0,
#                 "severity": "low",
#                 "domain_reputation": {
#                     "risk_level": "low",
#                     "risk_score": 0,
#                     "reasons": [],
#                     "is_free_provider": False,
#                     "domain": result.get("from_email", "unknown")
#                 }
#             }
        
#         return result
        
#     except Exception as e:
#         logger.exception("score_email failed")
#         raise HTTPException(status_code=500, detail=str(e))


# # ----------------------------------------------------------
# # SCORE BATCH
# # ----------------------------------------------------------
# @app.post("/score_batch")
# def score_batch(payload: BatchRequest):
#     """
#     Score multiple emails in batch
#     """
#     try:
#         results = []
#         for email in payload.emails:
#             try:
#                 r = predictor.predict_email(email.dict())
                
#                 # Optionally add Groq explanation for each
#                 if explainer.is_available():
#                     try:
#                         explanation = explainer.explain({
#                             "from": email.from_email or email.dict().get("from", ""),
#                             "subject": email.subject,
#                             "body": email.body,
#                             "model_prediction": r["predicted"],
#                             "risk_analysis": r.get("risk_analysis", {}),
#                             "reasons": r.get("reasons", [])
#                         })
#                         r["explanation"] = explanation
#                     except:
#                         pass  # Skip explanation on error
                
#                 results.append(r)
#             except Exception as e:
#                 logger.error(f"Failed to process email: {e}")
#                 results.append({
#                     "error": str(e),
#                     "message_id": email.dict().get("message_id", "unknown")
#                 })
        
#         return {"count": len(results), "results": results}
        
#     except Exception as e:
#         logger.exception("score_batch failed")
#         raise HTTPException(status_code=500, detail=str(e))


# # ----------------------------------------------------------
# # ANALYZE LINKS ENDPOINT
# # ----------------------------------------------------------
# @app.post("/analyze_links")
# def analyze_links(payload: EmailRequest):
#     """
#     Extract and analyze URLs in email
#     """
#     try:
#         import re
#         body = payload.body or ""
        
#         # Extract URLs
#         urls = re.findall(r"https?://\S+", body)
        
#         # Analyze each URL
#         url_analysis = []
#         for url in urls:
#             analysis = {
#                 "url": url,
#                 "is_https": url.startswith("https"),
#                 "has_ip": bool(re.search(r"https?://\d+\.\d+\.\d+\.\d+", url)),
#                 "suspicious_tld": any(url.endswith(tld) for tld in 
#                                      [".xyz", ".top", ".ru", ".cn", ".tk", ".ml"]),
#                 "is_shortened": any(short in url for short in 
#                                    ["bit.ly", "tinyurl", "t.co", "goo.gl"]),
#                 "has_hex_encoding": "%" in url,
#                 "length": len(url)
#             }
#             url_analysis.append(analysis)
        
#         # Calculate overall risk
#         high_risk_count = sum(1 for u in url_analysis 
#                              if u["has_ip"] or u["suspicious_tld"] or u["is_shortened"])
        
#         return {
#             "url_count": len(urls),
#             "urls": url_analysis,
#             "high_risk_urls": high_risk_count,
#             "risk_level": "high" if high_risk_count > 0 else "low"
#         }
        
#     except Exception as e:
#         logger.exception("analyze_links failed")
#         raise HTTPException(status_code=500, detail=str(e))


# # ----------------------------------------------------------
# # RISK REPORT (alias for score_email)
# # ----------------------------------------------------------
# @app.post("/risk_report", response_model=EmailClassificationResponse)
# def risk_report(payload: EmailRequest):
#     """
#     Generate comprehensive risk report for email
#     (Alias for score_email with same enhanced analysis)
#     """
#     return score_email(payload)


# # ----------------------------------------------------------
# # NEW: Get Risk Score Only
# # ----------------------------------------------------------
# @app.post("/risk_score")
# def risk_score(payload: EmailRequest):
#     """
#     Get just the risk score and severity (lightweight endpoint)
#     """
#     try:
#         result = predictor.predict_email(payload.dict())
        
#         return {
#             "message_id": result.get("message_id"),
#             "predicted_label": result["predicted"]["predicted_label"],
#             "risk_score": result.get("risk_analysis", {}).get("risk_score", 0),
#             "severity": result.get("risk_analysis", {}).get("severity", "low"),
#             "confidence": result["predicted"]["model_confidence"]
#         }
        
#     except Exception as e:
#         logger.exception("risk_score failed")
#         raise HTTPException(status_code=500, detail=str(e))


# # ----------------------------------------------------------
# # NEW: Domain Reputation Check
# # ----------------------------------------------------------
# @app.get("/domain_reputation/{domain}")
# def check_domain_reputation(domain: str):
#     """
#     Check reputation of a specific domain
#     """
#     try:
#         from predictor import analyze_domain_reputation
        
#         # Create fake email to analyze domain
#         fake_email = f"test@{domain}"
#         analysis = analyze_domain_reputation(fake_email)
        
#         return {
#             "domain": domain,
#             "reputation": analysis
#         }
        
#     except Exception as e:
#         logger.exception("domain_reputation check failed")
#         raise HTTPException(status_code=500, detail=str(e))









# from fastapi import FastAPI, HTTPException
# from fastapi.middleware.cors import CORSMiddleware
# import os
# import sys
# import logging


# # Add parent directory to path for database imports
# sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# from predictor import get_predictor
# from groq_explainer import GroqExplainer
# from schemas import EmailRequest, BatchRequest
# from response_models import EmailClassificationResponse
# from database.repository import get_repository

# logging.basicConfig(level=logging.INFO)
# logger = logging.getLogger("api.main")

# app = FastAPI(
#     title="Email Trust & Safety API (v3 Enhanced + PostgreSQL)",
#     description="Enterprise anti-abuse API with enhanced detection and PostgreSQL storage",
#     version="3.0.0-postgres",
# )

# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["*"],
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )

# # Initialize services
# predictor = get_predictor()
# explainer = GroqExplainer(api_key=os.getenv("GROQ_API_KEY"))
# repository = get_repository()


# @app.on_event("startup")
# async def startup_event():
#     """Initialize database on startup"""
#     logger.info("🚀 Starting Email Trust & Safety API")
#     logger.info("📊 PostgreSQL integration enabled")


# @app.get("/")
# def read_root():
#     return {
#         "service": "Email Trust & Safety API",
#         "version": "3.0.0-postgres",
#         "status": "running",
#         "model": "xgboost_v3_enhanced",
#         "database": "postgresql",
#         "features": [
#             "Multi-factor risk analysis",
#             "Domain reputation scoring",
#             "Enhanced phishing detection",
#             "40+ spam keyword detection",
#             "Aggressive auto-classification",
#             "PostgreSQL data storage",
#             "Real-time abuse tracking",
#             "Campaign detection"
#         ]
#     }


# @app.get("/health")
# def health():
#     return {
#         "status": "ok",
#         "model": "xgboost_v3_enhanced",
#         "database": "connected",
#         "features_loaded": True,
#         "groq_available": explainer.is_available()
#     }


# # ----------------------------------------------------------
# # SCORE SINGLE EMAIL (Enhanced with Database Storage)
# # ----------------------------------------------------------
# @app.post("/score_email", response_model=EmailClassificationResponse)
# def score_email(payload: EmailRequest):
#     """
#     Score a single email with enhanced risk analysis and store in PostgreSQL
#     """
#     try:
#         # Convert to dict for predictor
#         email_dict = payload.dict()
        
#         # Get prediction with enhanced risk analysis
#         result = predictor.predict_email(email_dict)
        
#         # ✅ STORE IN DATABASE
#         try:
#             stored_email = repository.store_classification(result)
#             logger.info(f"✓ Stored classification in database: ID {stored_email.id}")
#         except Exception as db_error:
#             logger.error(f"Database storage failed: {db_error}")
#             # Continue even if storage fails
        
#         # Add Groq explanation if available
#         if explainer.is_available():
#             try:
#                 explanation = explainer.explain({
#                     "from": payload.from_email or payload.dict().get("from", ""),
#                     "subject": payload.subject,
#                     "body": payload.body,
#                     "model_prediction": result["predicted"],
#                     "risk_analysis": result.get("risk_analysis", {}),
#                     "reasons": result.get("reasons", [])
#                 })
#                 result["explanation"] = explanation
#             except Exception as e:
#                 logger.warning(f"Groq explanation failed: {e}")
        
#         # Ensure risk_analysis exists
#         if "risk_analysis" not in result:
#             result["risk_analysis"] = {
#                 "risk_score": 0,
#                 "severity": "low",
#                 "domain_reputation": {
#                     "risk_level": "low",
#                     "risk_score": 0,
#                     "reasons": [],
#                     "is_free_provider": False,
#                     "domain": result.get("from_email", "unknown")
#                 }
#             }
        
#         return result
        
#     except Exception as e:
#         logger.exception("score_email failed")
#         raise HTTPException(status_code=500, detail=str(e))


# # ----------------------------------------------------------
# # SCORE BATCH
# # ----------------------------------------------------------
# @app.post("/score_batch")
# def score_batch(payload: BatchRequest):
#     """Score multiple emails in batch with database storage"""
#     try:
#         results = []
#         for email in payload.emails:
#             try:
#                 r = predictor.predict_email(email.dict())
                
#                 # Store in database
#                 try:
#                     repository.store_classification(r)
#                 except Exception as db_error:
#                     logger.error(f"Batch DB storage failed: {db_error}")
                
#                 results.append(r)
#             except Exception as e:
#                 logger.error(f"Failed to process email: {e}")
#                 results.append({
#                     "error": str(e),
#                     "message_id": email.dict().get("message_id", "unknown")
#                 })
        
#         return {"count": len(results), "results": results}
        
#     except Exception as e:
#         logger.exception("score_batch failed")
#         raise HTTPException(status_code=500, detail=str(e))


# # ----------------------------------------------------------
# # DATABASE QUERY ENDPOINTS
# # ----------------------------------------------------------
# @app.get("/stats/classification")
# def get_classification_stats(hours: int = 24):
#     """Get classification statistics from database"""
#     try:
#         stats = repository.get_classification_stats(hours)
#         return {
#             "time_period_hours": hours,
#             "statistics": stats
#         }
#     except Exception as e:
#         logger.exception("Failed to get stats")
#         raise HTTPException(status_code=500, detail=str(e))


# @app.get("/stats/domains/malicious")
# def get_malicious_domains(limit: int = 10):
#     """Get top malicious domains from database"""
#     try:
#         domains = repository.get_top_malicious_domains(limit)
#         return {
#             "domains": [
#                 {
#                     "domain": d.domain,
#                     "total_emails": d.total_emails,
#                     "phishing_count": d.phishing_count,
#                     "spam_count": d.spam_count,
#                     "risk_level": d.risk_level,
#                     "first_seen": d.first_seen.isoformat(),
#                     "last_seen": d.last_seen.isoformat()
#                 }
#                 for d in domains
#             ]
#         }
#     except Exception as e:
#         logger.exception("Failed to get malicious domains")
#         raise HTTPException(status_code=500, detail=str(e))


# @app.get("/stats/threats/recent")
# def get_recent_threats(limit: int = 20):
#     """Get recent high-risk emails from database"""
#     try:
#         threats = repository.get_recent_threats(limit)
#         return {
#             "threats": [
#                 {
#                     "message_id": t.message_id,
#                     "from_email": t.from_email,
#                     "subject": t.subject,
#                     "predicted_label": t.predicted_label,
#                     "risk_score": t.risk_score,
#                     "severity": t.severity,
#                     "classified_at": t.classified_at.isoformat()
#                 }
#                 for t in threats
#             ]
#         }
#     except Exception as e:
#         logger.exception("Failed to get recent threats")
#         raise HTTPException(status_code=500, detail=str(e))


# @app.get("/stats/campaigns/active")
# def get_active_campaigns():
#     """Get active abuse campaigns from database"""
#     try:
#         campaigns = repository.get_active_campaigns()
#         return {
#             "campaigns": [
#                 {
#                     "id": c.id,
#                     "name": c.campaign_name,
#                     "domain": c.domain,
#                     "attack_type": c.attack_type,
#                     "email_count": c.email_count,
#                     "severity": c.severity,
#                     "first_detected": c.first_detected.isoformat(),
#                     "last_detected": c.last_detected.isoformat()
#                 }
#                 for c in campaigns
#             ]
#         }
#     except Exception as e:
#         logger.exception("Failed to get campaigns")
#         raise HTTPException(status_code=500, detail=str(e))


# @app.get("/stats/volume/hourly")
# def get_hourly_volume(hours: int = 24):
#     """Get hourly email volume from database"""
#     try:
#         volume = repository.get_hourly_volume(hours)
#         return {
#             "time_period_hours": hours,
#             "hourly_data": volume
#         }
#     except Exception as e:
#         logger.exception("Failed to get hourly volume")
#         raise HTTPException(status_code=500, detail=str(e))


# @app.get("/sender/{email_address}/reputation")
# def get_sender_reputation(email_address: str):
#     """Get sender's reputation from database"""
#     try:
#         reputation = repository.get_sender_reputation(email_address)
#         return {
#             "email_address": email_address,
#             "reputation": reputation
#         }
#     except Exception as e:
#         logger.exception("Failed to get sender reputation")
#         raise HTTPException(status_code=500, detail=str(e))


# @app.get("/search/emails")
# def search_emails(
#     from_email: str = None,
#     label: str = None,
#     min_risk: int = None,
#     limit: int = 100
# ):
#     """Search emails in database with filters"""
#     try:
#         emails = repository.search_emails(from_email, label, min_risk, limit)
#         return {
#             "count": len(emails),
#             "emails": [
#                 {
#                     "message_id": e.message_id,
#                     "from_email": e.from_email,
#                     "subject": e.subject,
#                     "predicted_label": e.predicted_label,
#                     "risk_score": e.risk_score,
#                     "classified_at": e.classified_at.isoformat()
#                 }
#                 for e in emails
#             ]
#         }
#     except Exception as e:
#         logger.exception("Search failed")
#         raise HTTPException(status_code=500, detail=str(e))


# # ----------------------------------------------------------
# # ANALYZE LINKS ENDPOINT
# # ----------------------------------------------------------
# @app.post("/analyze_links")
# def analyze_links(payload: EmailRequest):
#     """Extract and analyze URLs in email"""
#     try:
#         import re
#         body = payload.body or ""
        
#         urls = re.findall(r"https?://\S+", body)
        
#         url_analysis = []
#         for url in urls:
#             analysis = {
#                 "url": url,
#                 "is_https": url.startswith("https"),
#                 "has_ip": bool(re.search(r"https?://\d+\.\d+\.\d+\.\d+", url)),
#                 "suspicious_tld": any(url.endswith(tld) for tld in 
#                                      [".xyz", ".top", ".ru", ".cn", ".tk", ".ml"]),
#                 "is_shortened": any(short in url for short in 
#                                    ["bit.ly", "tinyurl", "t.co", "goo.gl"]),
#                 "has_hex_encoding": "%" in url,
#                 "length": len(url)
#             }
#             url_analysis.append(analysis)
        
#         high_risk_count = sum(1 for u in url_analysis 
#                              if u["has_ip"] or u["suspicious_tld"] or u["is_shortened"])
        
#         return {
#             "url_count": len(urls),
#             "urls": url_analysis,
#             "high_risk_urls": high_risk_count,
#             "risk_level": "high" if high_risk_count > 0 else "low"
#         }
        
#     except Exception as e:
#         logger.exception("analyze_links failed")
#         raise HTTPException(status_code=500, detail=str(e))


from dotenv import load_dotenv
load_dotenv()


from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import os
import sys
import logging

# Add parent directory to path for database imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from predictor import get_predictor
from groq_explainer import GroqExplainer
from schemas import EmailRequest, BatchRequest
from response_models import EmailClassificationResponse

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("api.main")

app = FastAPI(
    title="Email Trust & Safety API (v3 Enhanced + PostgreSQL)",
    description="Enterprise anti-abuse API with enhanced detection and PostgreSQL storage",
    version="3.0.0-postgres",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize services
predictor = get_predictor()
explainer = GroqExplainer(api_key=os.getenv("GROQ_API_KEY"))

# Database repository (will be initialized on startup)
repository = None


def setup_database():
    """Setup database automatically on startup"""
    try:
        logger.info("🔧 Setting up PostgreSQL database...")
        
        # Check if database package exists
        try:
            from database.connection import get_db_manager, init_database
            from database.repository import get_repository
        except ImportError as e:
            logger.warning(f"⚠️ Database module not found: {e}")
            logger.warning("📝 Running without database storage")
            logger.warning("   To enable database: Install packages and create database/ folder")
            return None
        
        # Check environment variables
        db_user = os.getenv('DB_USER', 'postgres')
        db_password = os.getenv('DB_PASSWORD')
        db_host = os.getenv('DB_HOST', 'localhost')
        db_name = os.getenv('DB_NAME', 'email_trust_safety')
        
        if not db_password:
            logger.warning("⚠️ DB_PASSWORD not set in environment")
            logger.warning("   Set DB_PASSWORD in .env file to enable database")
            return None
        
        # Try to connect and initialize
        try:
            # Create database if it doesn't exist
            import psycopg2
            from psycopg2.extensions import ISOLATION_LEVEL_AUTOCOMMIT
            
            try:
                conn = psycopg2.connect(
                    dbname='postgres',
                    user=db_user,
                    password=db_password,
                    host=db_host
                )
                conn.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)
                cursor = conn.cursor()
                
                # Check if database exists
                cursor.execute("SELECT 1 FROM pg_database WHERE datname = %s", (db_name,))
                if not cursor.fetchone():
                    cursor.execute(f'CREATE DATABASE {db_name}')
                    logger.info(f"✓ Created database '{db_name}'")
                else:
                    logger.info(f"✓ Database '{db_name}' exists")
                
                cursor.close()
                conn.close()
            except psycopg2.Error as e:
                logger.error(f"✗ Database connection failed: {e}")
                return None
            
            # Initialize tables
            init_database()
            logger.info("✓ Database tables initialized")
            
            # Get repository
            repo = get_repository()
            logger.info("✓ Database repository ready")
            return repo
            
        except Exception as e:
            logger.error(f"✗ Database initialization failed: {e}")
            logger.warning("📝 Running without database storage")
            return None
            
    except Exception as e:
        logger.error(f"✗ Database setup error: {e}")
        logger.warning("📝 Running without database storage")
        return None


@app.on_event("startup")
async def startup_event():
    """Initialize services on startup"""
    global repository
    
    logger.info("=" * 60)
    logger.info("🚀 Starting Email Trust & Safety API")
    logger.info("=" * 60)
    
    # Try to setup database
    repository = setup_database()
    
    if repository:
        logger.info("✓ PostgreSQL integration ENABLED")
    else:
        logger.info("ℹ️ PostgreSQL integration DISABLED")
        logger.info("   API will work but classifications won't be stored")
    
    logger.info("=" * 60)


@app.get("/")
def read_root():
    return {
        "service": "Email Trust & Safety API",
        "version": "3.0.0-postgres",
        "status": "running",
        "model": "xgboost_v3_enhanced",
        "database": "enabled" if repository else "disabled",
        "features": [
            "Multi-factor risk analysis",
            "Domain reputation scoring",
            "Enhanced phishing detection",
            "40+ spam keyword detection",
            "Aggressive auto-classification",
            "PostgreSQL data storage (optional)",
            "Real-time abuse tracking",
            "Campaign detection"
        ]
    }


@app.get("/health")
def health():
    return {
        "status": "ok",
        "model": "xgboost_v3_enhanced",
        "database": "connected" if repository else "disabled",
        "features_loaded": True,
        "groq_available": explainer.is_available()
    }


# ----------------------------------------------------------
# SCORE SINGLE EMAIL (Enhanced with Optional Database Storage)
# ----------------------------------------------------------
@app.post("/score_email", response_model=EmailClassificationResponse)
def score_email(payload: EmailRequest):
    """
    Score a single email with enhanced risk analysis
    Stores in PostgreSQL if available
    """
    try:
        # Convert to dict for predictor
        email_dict = payload.dict()
        
        # Get prediction with enhanced risk analysis
        result = predictor.predict_email(email_dict)
        
        # ✅ STORE IN DATABASE (if available)
        if repository:
            try:
                stored_email = repository.store_classification(result)
                logger.info(f"✓ Stored in DB: ID {stored_email.id}")
            except Exception as db_error:
                logger.error(f"Database storage failed: {db_error}")
                # Continue even if storage fails
        
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
        
        # Ensure risk_analysis exists
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
    """Score multiple emails in batch with optional database storage"""
    try:
        results = []
        for email in payload.emails:
            try:
                r = predictor.predict_email(email.dict())
                
                # Store in database if available
                if repository:
                    try:
                        repository.store_classification(r)
                    except Exception as db_error:
                        logger.error(f"Batch DB storage failed: {db_error}")
                
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
# DATABASE QUERY ENDPOINTS (Only work if database is enabled)
# ----------------------------------------------------------
@app.get("/stats/classification")
def get_classification_stats(hours: int = 24):
    """Get classification statistics from database"""
    if not repository:
        raise HTTPException(status_code=503, detail="Database not available")
    
    try:
        stats = repository.get_classification_stats(hours)
        return {
            "time_period_hours": hours,
            "statistics": stats
        }
    except Exception as e:
        logger.exception("Failed to get stats")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/stats/domains/malicious")
def get_malicious_domains(limit: int = 10):
    """Get top malicious domains from database"""
    if not repository:
        raise HTTPException(status_code=503, detail="Database not available")
    
    try:
        domains = repository.get_top_malicious_domains(limit)
        return {
            "domains": [
                {
                    "domain": d.domain,
                    "total_emails": d.total_emails,
                    "phishing_count": d.phishing_count,
                    "spam_count": d.spam_count,
                    "risk_level": d.risk_level,
                    "first_seen": d.first_seen.isoformat(),
                    "last_seen": d.last_seen.isoformat()
                }
                for d in domains
            ]
        }
    except Exception as e:
        logger.exception("Failed to get malicious domains")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/stats/threats/recent")
def get_recent_threats(limit: int = 20):
    """Get recent high-risk emails from database"""
    if not repository:
        raise HTTPException(status_code=503, detail="Database not available")
    
    try:
        threats = repository.get_recent_threats(limit)
        return {
            "threats": [
                {
                    "message_id": t.message_id,
                    "from_email": t.from_email,
                    "subject": t.subject,
                    "predicted_label": t.predicted_label,
                    "risk_score": t.risk_score,
                    "severity": t.severity,
                    "classified_at": t.classified_at.isoformat()
                }
                for t in threats
            ]
        }
    except Exception as e:
        logger.exception("Failed to get recent threats")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/stats/campaigns/active")
def get_active_campaigns():
    """Get active abuse campaigns from database"""
    if not repository:
        raise HTTPException(status_code=503, detail="Database not available")
    
    try:
        campaigns = repository.get_active_campaigns()
        return {
            "campaigns": [
                {
                    "id": c.id,
                    "name": c.campaign_name,
                    "domain": c.domain,
                    "attack_type": c.attack_type,
                    "email_count": c.email_count,
                    "severity": c.severity,
                    "first_detected": c.first_detected.isoformat(),
                    "last_detected": c.last_detected.isoformat()
                }
                for c in campaigns
            ]
        }
    except Exception as e:
        logger.exception("Failed to get campaigns")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/stats/volume/hourly")
def get_hourly_volume(hours: int = 24):
    """Get hourly email volume from database"""
    if not repository:
        raise HTTPException(status_code=503, detail="Database not available")
    
    try:
        volume = repository.get_hourly_volume(hours)
        return {
            "time_period_hours": hours,
            "hourly_data": volume
        }
    except Exception as e:
        logger.exception("Failed to get hourly volume")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/sender/{email_address}/reputation")
def get_sender_reputation(email_address: str):
    """Get sender's reputation from database"""
    if not repository:
        raise HTTPException(status_code=503, detail="Database not available")
    
    try:
        reputation = repository.get_sender_reputation(email_address)
        return {
            "email_address": email_address,
            "reputation": reputation
        }
    except Exception as e:
        logger.exception("Failed to get sender reputation")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/search/emails")
def search_emails(
    from_email: str = None,
    label: str = None,
    min_risk: int = None,
    limit: int = 100
):
    """Search emails in database with filters"""
    if not repository:
        raise HTTPException(status_code=503, detail="Database not available")
    
    try:
        emails = repository.search_emails(from_email, label, min_risk, limit)
        return {
            "count": len(emails),
            "emails": [
                {
                    "message_id": e.message_id,
                    "from_email": e.from_email,
                    "subject": e.subject,
                    "predicted_label": e.predicted_label,
                    "risk_score": e.risk_score,
                    "classified_at": e.classified_at.isoformat()
                }
                for e in emails
            ]
        }
    except Exception as e:
        logger.exception("Search failed")
        raise HTTPException(status_code=500, detail=str(e))


# ----------------------------------------------------------
# ANALYZE LINKS ENDPOINT
# ----------------------------------------------------------
@app.post("/analyze_links")
def analyze_links(payload: EmailRequest):
    """Extract and analyze URLs in email"""
    try:
        import re
        body = payload.body or ""
        
        urls = re.findall(r"https?://\S+", body)
        
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


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)