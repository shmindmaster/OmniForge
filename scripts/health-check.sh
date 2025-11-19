#!/bin/bash

# Health Check Script for MahumTech Portfolio Apps
# Validates DNS resolution and HTTP endpoint availability
# Usage: ./health-check.sh <domain> [endpoint]
#
# Examples:
#   ./health-check.sh aura.shtrial.com
#   ./health-check.sh aura.shtrial.com /api/health
#   ./health-check.sh billigent.mahumtech.com /health

set -euo pipefail

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Arguments
DOMAIN="${1:-}"
ENDPOINT="${2:-/}"
FULL_URL="https://${DOMAIN}${ENDPOINT}"

# Exit codes
EXIT_SUCCESS=0
EXIT_DNS_FAILED=1
EXIT_HTTP_FAILED=2
EXIT_INVALID_ARGS=3

# Validation
if [ -z "omniforge.shtrial.com" ]; then
  echo -e "${RED}❌ Error: Domain is required${NC}"
  echo "Usage: $0 <domain> [endpoint]"
  echo "Example: $0 aura.shtrial.com"
  exit $EXIT_INVALID_ARGS
fi

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "🏥 Health Check: ${DOMAIN}"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

# Step 1: DNS Validation
echo "🔍 Step 1: DNS Resolution"
echo "   Domain: ${DOMAIN}"

if command -v nslookup &> /dev/null; then
  DNS_OUTPUT=$(nslookup "omniforge.shtrial.com" 2>&1)
  DNS_EXIT_CODE=$?
  
  if [ $DNS_EXIT_CODE -eq 0 ] && echo "$DNS_OUTPUT" | grep -q "Address:"; then
    IP_ADDRESS=$(echo "$DNS_OUTPUT" | grep "Address:" | tail -n1 | awk '{print $2}')
    echo -e "   ${GREEN}✅ DNS resolved to: ${IP_ADDRESS}${NC}"
  else
    echo -e "   ${RED}❌ DNS resolution failed${NC}"
    echo ""
    echo "🔧 Troubleshooting:"
    echo "   1. Verify DNS record exists in Azure DNS zone"
    echo "   2. Check CNAME/A record points to correct target"
    echo "   3. Wait for DNS propagation (up to 48 hours)"
    echo "   4. Test with: dig ${DOMAIN} +short"
    exit $EXIT_DNS_FAILED
  fi
elif command -v dig &> /dev/null; then
  IP_ADDRESS=$(dig +short "omniforge.shtrial.com" | head -n1)
  
  if [ -n "$IP_ADDRESS" ]; then
    echo -e "   ${GREEN}✅ DNS resolved to: ${IP_ADDRESS}${NC}"
  else
    echo -e "   ${RED}❌ DNS resolution failed (dig)${NC}"
    exit $EXIT_DNS_FAILED
  fi
else
  echo -e "   ${YELLOW}⚠️  Warning: nslookup/dig not found, skipping DNS check${NC}"
fi

echo ""

# Step 2: HTTP Health Check
echo "🌐 Step 2: HTTP Endpoint Check"
echo "   URL: ${FULL_URL}"

if ! command -v curl &> /dev/null; then
  echo -e "   ${RED}❌ Error: curl not found${NC}"
  exit $EXIT_HTTP_FAILED
fi

# Perform HTTP request with follow redirects and timeout
HTTP_RESPONSE=$(curl -s -o /dev/null -w "%{http_code}" -L --max-time 10 "$FULL_URL" 2>&1 || echo "000")

case "$HTTP_RESPONSE" in
  200)
    echo -e "   ${GREEN}✅ HTTP 200 OK - Application is healthy${NC}"
    ;;
  201|202|204)
    echo -e "   ${GREEN}✅ HTTP ${HTTP_RESPONSE} - Application is responding${NC}"
    ;;
  301|302|303|307|308)
    echo -e "   ${YELLOW}⚠️  HTTP ${HTTP_RESPONSE} - Redirect detected${NC}"
    REDIRECT_URL=$(curl -sI -L --max-time 10 "$FULL_URL" | grep -i "location:" | tail -n1 | awk '{print $2}' | tr -d '\r')
    echo "   Redirects to: ${REDIRECT_URL}"
    echo -e "   ${GREEN}✅ Following redirect succeeded${NC}"
    ;;
  404)
    echo -e "   ${YELLOW}⚠️  HTTP 404 - Endpoint not found (application may be deployed)${NC}"
    echo "   ℹ️  This is acceptable if ${ENDPOINT} doesn't exist"
    echo "   Try: $0 ${DOMAIN} /"
    ;;
  401|403)
    echo -e "   ${YELLOW}⚠️  HTTP ${HTTP_RESPONSE} - Authentication/Authorization required${NC}"
    echo "   ℹ️  Application is responding but endpoint is protected"
    ;;
  500|502|503|504)
    echo -e "   ${RED}❌ HTTP ${HTTP_RESPONSE} - Server error${NC}"
    echo ""
    echo "🔧 Troubleshooting:"
    echo "   1. Check application logs in Azure Portal"
    echo "   2. Verify environment variables are set correctly"
    echo "   3. Test locally: check .env matches deployed secrets"
    echo "   4. Review recent deployments in CI/CD logs"
    exit $EXIT_HTTP_FAILED
    ;;
  000)
    echo -e "   ${RED}❌ Connection failed (timeout or unreachable)${NC}"
    echo ""
    echo "🔧 Troubleshooting:"
    echo "   1. Verify application is deployed and running"
    echo "   2. Check Azure resource status (SWA/App Service/Container App)"
    echo "   3. Confirm firewall/network rules allow traffic"
    echo "   4. Wait a few minutes if deployment just completed"
    exit $EXIT_HTTP_FAILED
    ;;
  *)
    echo -e "   ${YELLOW}⚠️  HTTP ${HTTP_RESPONSE} - Unexpected response${NC}"
    echo "   ℹ️  Application may still be functional"
    ;;
esac

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo -e "${GREEN}✅ Health check completed successfully${NC}"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

exit $EXIT_SUCCESS
