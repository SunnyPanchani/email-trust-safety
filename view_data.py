#!/usr/bin/env python3
"""
View Email Classification Data from PostgreSQL
Run this script to see your stored data
"""

import os
import sys
from datetime import datetime, timedelta
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from database.repository import get_repository
from database.connection import get_db_manager
from database.models import Email, DomainReputation, AbuseCampaign

def print_separator():
    print("=" * 80)

def view_recent_emails(limit=10):
    """View most recent classified emails"""
    print_separator()
    print(f"📧 RECENT {limit} EMAILS")
    print_separator()
    
    repo = get_repository()
    db = get_db_manager()
    
    with db.get_session() as session:
        emails = session.query(Email).order_by(Email.classified_at.desc()).limit(limit).all()
        
        if not emails:
            print("❌ No emails found in database")
            print("\nTo add test data, send a POST request to:")
            print("   http://localhost:8000/score_email")
            return
        
        for i, email in enumerate(emails, 1):
            print(f"\n{i}. Email ID: {email.id}")
            print(f"   From: {email.from_email}")
            print(f"   Subject: {email.subject[:60]}...")
            print(f"   Classification: {email.predicted_label.upper()}")
            print(f"   Risk Score: {email.risk_score}/100 ({email.severity})")
            print(f"   Confidence: {email.model_confidence:.2%}")
            print(f"   Time: {email.classified_at.strftime('%Y-%m-%d %H:%M:%S')}")

def view_classification_stats():
    """View classification statistics"""
    print_separator()
    print("📊 CLASSIFICATION STATISTICS (Last 24 Hours)")
    print_separator()
    
    repo = get_repository()
    stats = repo.get_classification_stats(hours=24)
    
    if not stats:
        print("❌ No classifications in last 24 hours")
        return
    
    total = sum(stats.values())
    print(f"\nTotal Emails: {total}")
    print("-" * 40)
    
    for label, count in stats.items():
        percentage = (count / total * 100) if total > 0 else 0
        bar = "█" * int(percentage / 2)
        print(f"{label.upper():12} : {count:4} ({percentage:5.1f}%) {bar}")

def view_malicious_domains(limit=10):
    """View top malicious domains"""
    print_separator()
    print(f"🚨 TOP {limit} MALICIOUS DOMAINS")
    print_separator()
    
    repo = get_repository()
    domains = repo.get_top_malicious_domains(limit=limit)
    
    if not domains:
        print("✅ No malicious domains detected yet")
        return
    
    for i, domain in enumerate(domains, 1):
        print(f"\n{i}. {domain.domain}")
        print(f"   Risk Level: {domain.risk_level.upper()}")
        print(f"   Total Emails: {domain.total_emails}")
        print(f"   Phishing: {domain.phishing_count}")
        print(f"   Spam: {domain.spam_count}")
        print(f"   Ham: {domain.ham_count}")
        print(f"   First Seen: {domain.first_seen.strftime('%Y-%m-%d')}")

def view_high_risk_threats(limit=10):
    """View high-risk emails"""
    print_separator()
    print(f"⚠️  HIGH-RISK THREATS (Risk Score >= 60)")
    print_separator()
    
    repo = get_repository()
    threats = repo.get_recent_threats(limit=limit)
    
    if not threats:
        print("✅ No high-risk threats detected")
        return
    
    for i, threat in enumerate(threats, 1):
        print(f"\n{i}. Threat ID: {threat.id}")
        print(f"   From: {threat.from_email}")
        print(f"   Subject: {threat.subject[:60]}...")
        print(f"   Type: {threat.predicted_label.upper()}")
        print(f"   Risk Score: {threat.risk_score}/100 ({threat.severity})")
        print(f"   Detected: {threat.classified_at.strftime('%Y-%m-%d %H:%M:%S')}")

def view_active_campaigns():
    """View active abuse campaigns"""
    print_separator()
    print("🎯 ACTIVE ABUSE CAMPAIGNS")
    print_separator()
    
    repo = get_repository()
    campaigns = repo.get_active_campaigns()
    
    if not campaigns:
        print("✅ No active campaigns detected")
        return
    
    for i, campaign in enumerate(campaigns, 1):
        print(f"\n{i}. Campaign: {campaign.campaign_name}")
        print(f"   Domain: {campaign.domain}")
        print(f"   Type: {campaign.attack_type}")
        print(f"   Emails: {campaign.email_count}")
        print(f"   Severity: {campaign.severity.upper()}")
        print(f"   First Detected: {campaign.first_detected.strftime('%Y-%m-%d %H:%M')}")
        print(f"   Last Detected: {campaign.last_detected.strftime('%Y-%m-%d %H:%M')}")

def view_database_info():
    """View database connection info"""
    print_separator()
    print("🗄️  DATABASE INFORMATION")
    print_separator()
    
    print(f"\nDatabase Name: {os.getenv('DB_NAME', 'email_trust_safety')}")
    print(f"Host: {os.getenv('DB_HOST', 'localhost')}")
    print(f"Port: {os.getenv('DB_PORT', '5432')}")
    print(f"User: {os.getenv('DB_USER', 'postgres')}")
    
    db = get_db_manager()
    
    with db.get_session() as session:
        email_count = session.query(Email).count()
        domain_count = session.query(DomainReputation).count()
        campaign_count = session.query(AbuseCampaign).count()
        
        print(f"\n📊 Table Counts:")
        print(f"   Emails: {email_count}")
        print(f"   Domains Tracked: {domain_count}")
        print(f"   Campaigns: {campaign_count}")

def main():
    """Main function"""
    print("\n")
    print("╔" + "═" * 78 + "╗")
    print("║" + " " * 20 + "EMAIL TRUST & SAFETY DATABASE VIEWER" + " " * 22 + "║")
    print("╚" + "═" * 78 + "╝")
    print()
    
    try:
        # Initialize database connection
        db = get_db_manager()
        db.initialize()
        
        # Display data
        view_database_info()
        print()
        
        view_recent_emails(limit=10)
        print()
        
        view_classification_stats()
        print()
        
        view_high_risk_threats(limit=5)
        print()
        
        view_malicious_domains(limit=5)
        print()
        
        view_active_campaigns()
        print()
        
        print_separator()
        print("✅ Data retrieval complete!")
        print_separator()
        print()
        print("💡 TIP: To view more data, edit this script or use SQL queries")
        print("   SQL queries available in: view_postgres_data.sql artifact")
        print()
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        print("\nMake sure:")
        print("1. PostgreSQL is running")
        print("2. Database 'email_trust_safety' exists")
        print("3. .env file has correct credentials")
        print("4. Run: python setup_database.py")

if __name__ == "__main__":
    main()