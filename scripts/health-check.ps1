# Health Check Script for Portfolio Apps
# PowerShell version for Windows environments
# Usage: .\health-check.ps1 -Domain <domain> [-Endpoint <path>]
#
# Examples:
#   .\health-check.ps1 -Domain aura.shtrial.com
#   .\health-check.ps1 -Domain aura.shtrial.com -Endpoint "/api/health"
#   .\health-check.ps1 -Domain billigent.mahumtech.com -Endpoint "/health"

param(
    [Parameter(Mandatory=$true)]
    [string]$Domain,
    
    [Parameter(Mandatory=$false)]
    [string]$Endpoint = "/"
)

$ErrorActionPreference = "Stop"

# Exit codes
$EXIT_SUCCESS = 0
$EXIT_DNS_FAILED = 1
$EXIT_HTTP_FAILED = 2

# Full URL
$FullUrl = "https://${Domain}${Endpoint}"

Write-Host "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━" -ForegroundColor Gray
Write-Host "🏥 Health Check: $Domain" -ForegroundColor Cyan
Write-Host "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━" -ForegroundColor Gray
Write-Host ""

# Step 1: DNS Validation
Write-Host "🔍 Step 1: DNS Resolution" -ForegroundColor Yellow
Write-Host "   Domain: $Domain"

try {
    $dnsResult = Resolve-DnsName -Name $Domain -ErrorAction Stop
    $ipAddress = $dnsResult | Where-Object { $_.Type -eq 'A' } | Select-Object -First 1 -ExpandProperty IPAddress
    
    if ($ipAddress) {
        Write-Host "   ✅ DNS resolved to: $ipAddress" -ForegroundColor Green
    } else {
        $cnameTarget = $dnsResult | Where-Object { $_.Type -eq 'CNAME' } | Select-Object -First 1 -ExpandProperty NameHost
        if ($cnameTarget) {
            Write-Host "   ✅ DNS resolved (CNAME): $cnameTarget" -ForegroundColor Green
        } else {
            throw "No A or CNAME record found"
        }
    }
} catch {
    Write-Host "   ❌ DNS resolution failed" -ForegroundColor Red
    Write-Host ""
    Write-Host "🔧 Troubleshooting:" -ForegroundColor Yellow
    Write-Host "   1. Verify DNS record exists in Azure DNS zone"
    Write-Host "   2. Check CNAME/A record points to correct target"
    Write-Host "   3. Wait for DNS propagation (up to 48 hours)"
    Write-Host "   4. Test with: Resolve-DnsName $Domain"
    exit $EXIT_DNS_FAILED
}

Write-Host ""

# Step 2: HTTP Health Check
Write-Host "🌐 Step 2: HTTP Endpoint Check" -ForegroundColor Yellow
Write-Host "   URL: $FullUrl"

try {
    $response = Invoke-WebRequest -Uri $FullUrl -Method GET -TimeoutSec 10 -MaximumRedirection 5 -ErrorAction Stop
    $statusCode = $response.StatusCode
    
    switch ($statusCode) {
        200 {
            Write-Host "   ✅ HTTP 200 OK - Application is healthy" -ForegroundColor Green
        }
        { $_ -in 201, 202, 204 } {
            Write-Host "   ✅ HTTP $statusCode - Application is responding" -ForegroundColor Green
        }
        { $_ -in 301, 302, 303, 307, 308 } {
            Write-Host "   ⚠️  HTTP $statusCode - Redirect detected" -ForegroundColor Yellow
            Write-Host "   ✅ Following redirect succeeded" -ForegroundColor Green
        }
        default {
            Write-Host "   ⚠️  HTTP $statusCode - Unexpected response" -ForegroundColor Yellow
        }
    }
} catch {
    $statusCode = $_.Exception.Response.StatusCode.Value__
    
    switch ($statusCode) {
        404 {
            Write-Host "   ⚠️  HTTP 404 - Endpoint not found (application may be deployed)" -ForegroundColor Yellow
            Write-Host "   ℹ️  This is acceptable if $Endpoint doesn't exist"
            Write-Host "   Try: .\health-check.ps1 -Domain $Domain -Endpoint /"
        }
        { $_ -in 401, 403 } {
            Write-Host "   ⚠️  HTTP $statusCode - Authentication/Authorization required" -ForegroundColor Yellow
            Write-Host "   ℹ️  Application is responding but endpoint is protected"
        }
        { $_ -in 500, 502, 503, 504 } {
            Write-Host "   ❌ HTTP $statusCode - Server error" -ForegroundColor Red
            Write-Host ""
            Write-Host "🔧 Troubleshooting:" -ForegroundColor Yellow
            Write-Host "   1. Check application logs in Azure Portal"
            Write-Host "   2. Verify environment variables are set correctly"
            Write-Host "   3. Test locally: check .env matches deployed secrets"
            Write-Host "   4. Review recent deployments in CI/CD logs"
            exit $EXIT_HTTP_FAILED
        }
        default {
            if ($statusCode) {
                Write-Host "   ❌ HTTP $statusCode - Request failed" -ForegroundColor Red
            } else {
                Write-Host "   ❌ Connection failed (timeout or unreachable)" -ForegroundColor Red
            }
            Write-Host ""
            Write-Host "🔧 Troubleshooting:" -ForegroundColor Yellow
            Write-Host "   1. Verify application is deployed and running"
            Write-Host "   2. Check Azure resource status (SWA/App Service/Container App)"
            Write-Host "   3. Confirm firewall/network rules allow traffic"
            Write-Host "   4. Wait a few minutes if deployment just completed"
            exit $EXIT_HTTP_FAILED
        }
    }
}

Write-Host ""
Write-Host "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━" -ForegroundColor Gray
Write-Host "✅ Health check completed successfully" -ForegroundColor Green
Write-Host "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━" -ForegroundColor Gray

exit $EXIT_SUCCESS
