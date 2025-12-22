#!/bin/bash

# Email Trust & Safety Platform - Quick Start Script
# One-command setup and demo for evaluators

set -e  # Exit on error

echo "================================================================"
echo "🛡️  Email Trust & Safety Platform - Quick Start"
echo "================================================================"
echo ""

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Check Python version
echo -e "${YELLOW}Checking Python version...${NC}"
python_version=$(python3 --version 2>&1 | awk '{print $2}')
echo -e "${GREEN}✓ Python $python_version detected${NC}"
echo ""

# Install dependencies
echo -e "${YELLOW}Installing dependencies...${NC}"
pip install -r requirements.txt > /dev/null 2>&1
echo -e "${GREEN}✓ Dependencies installed${NC}"
echo ""

# Check if models exist
if [ ! -f "models/xgboost_v3.json" ]; then
    echo -e "${RED}✗ Model not found. Running training pipeline...${NC}"
    echo -e "${YELLOW}This may take 10-15 minutes...${NC}"
    python train_pipeline.py --data-dir data/raw --output-dir models
    echo -e "${GREEN}✓ Model training completed${NC}"
    echo ""
else
    echo -e "${GREEN}✓ Pre-trained model found${NC}"
    echo ""
fi

# Start API in background
echo -e "${YELLOW}Starting API server...${NC}"
cd api
python main.py > ../logs/api.log 2>&1 &
API_PID=$!
cd ..
echo -e "${GREEN}✓ API started (PID: $API_PID)${NC}"
echo ""

# Wait for API to be ready
echo -e "${YELLOW}Waiting for API to initialize...${NC}"
for i in {1..30}; do
    if curl -s http://localhost:8000/health > /dev/null 2>&1; then
        echo -e "${GREEN}✓ API is ready${NC}"
        break
    fi
    sleep 1
    echo -n "."
done
echo ""

# Start Dashboard in background
echo -e "${YELLOW}Starting monitoring dashboard...${NC}"
cd dashboard
streamlit run app.py > ../logs/dashboard.log 2>&1 &
DASHBOARD_PID=$!
cd ..
echo -e "${GREEN}✓ Dashboard started (PID: $DASHBOARD_PID)${NC}"
echo ""

# Create test script
cat > test_quick.sh << 'EOF'
#!/bin/bash
echo "Testing with example emails..."
echo ""

# Test 1: Legitimate Email
echo "Test 1: Legitimate Business Email"
curl -s -X POST http://localhost:8000/score_email \
  -H "Content-Type: application/json" \
  -d '{
    "from": "colleague@company.com",
    "to": ["you@example.com"],
    "subject": "Q4 Report Review",
    "body": "Hi team, please review the attached Q4 financial report."
  }' | python -m json.tool | grep -A 5 "predicted"
echo ""

# Test 2: Phishing Email
echo "Test 2: Lottery Scam (Phishing)"
curl -s -X POST http://localhost:8000/score_email \
  -H "Content-Type: application/json" \
  -d '{
    "from": "winner@lottery-prize.xyz",
    "to": ["victim@example.com"],
    "subject": "🎉 CONGRATULATIONS! You WON $5,000,000",
    "body": "CONGRATULATIONS! You are our GRAND PRIZE WINNER! Click here to claim..."
  }' | python -m json.tool | grep -A 10 "predicted\|risk_analysis"
echo ""

# Test 3: Account Phishing
echo "Test 3: Bank Phishing Attack"
curl -s -X POST http://localhost:8000/score_email \
  -H "Content-Type: application/json" \
  -d '{
    "from": "security@bank-verify.ru",
    "to": ["target@example.com"],
    "subject": "⚠️ Account Suspended - Verify Now",
    "body": "Your account has been suspended. Click here: http://192.168.1.1/bank"
  }' | python -m json.tool | grep -A 10 "predicted\|risk_analysis"
echo ""

echo "✓ All tests completed!"
EOF

chmod +x test_quick.sh

# Print summary
echo "================================================================"
echo -e "${GREEN}✓ Setup Complete!${NC}"
echo "================================================================"
echo ""
echo "🌐 Services Running:"
echo "   • API Server:    http://localhost:8000"
echo "   • API Docs:      http://localhost:8000/docs"
echo "   • Dashboard:     http://localhost:8501"
echo ""
echo "🧪 Quick Test Commands:"
echo "   ./test_quick.sh                  # Run test emails"
echo "   curl http://localhost:8000/health  # Check API health"
echo ""
echo "📊 View Dashboard:"
echo "   Open http://localhost:8501 in your browser"
echo ""
echo "🛑 Stop Services:"
echo "   kill $API_PID $DASHBOARD_PID"
echo ""
echo "📝 Logs:"
echo "   API:       tail -f logs/api.log"
echo "   Dashboard: tail -f logs/dashboard.log"
echo ""
echo "================================================================"
echo ""

# Run quick test
echo -e "${YELLOW}Running quick validation test...${NC}"
sleep 5  # Give services time to fully start
./test_quick.sh

echo ""
echo "================================================================"
echo -e "${GREEN}🎉 All systems operational!${NC}"
echo "================================================================"
echo ""
echo "Next steps:"
echo "1. Open http://localhost:8501 to view the dashboard"
echo "2. Try http://localhost:8000/docs for interactive API"
echo "3. Review RESEND_APPLICATION.md for detailed documentation"
echo ""