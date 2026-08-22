# hermes-install-plugin.ps1
# Install the openrouter-better-proxy Hermes plugin on Windows
#
# Usage:
#   .\hermes-install-plugin.ps1 [-ProxyUrl "http://192.168.1.251:8787/v1"]
#
# Parameters:
#   -ProxyUrl  Optional. URL of the proxy (default: http://localhost:8787/v1)
#
# Example:
#   .\hermes-install-plugin.ps1 -ProxyUrl "http://192.168.1.251:8787/v1"

param(
    [string]$ProxyUrl = "http://localhost:8787/v1"
)

$ErrorActionPreference = "Stop"

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$PluginSrc = Join-Path $ScriptDir "hermes-plugin\openrouter-better-proxy"
$HermesHome = [System.Environment]::GetFolderPath("LocalApplicationData") + "\hermes"
$PluginDir = Join-Path $HermesHome "plugins\model-providers\openrouter-better-proxy"

# Check if Hermes is installed
if (-not (Test-Path $HermesHome)) {
    Write-Host "ERROR: Hermes not found at $HermesHome" -ForegroundColor Red
    Write-Host "Please install Hermes first: https://hermes-agent.nousresearch.com/docs" -ForegroundColor Red
    exit 1
}

# Create plugin directory if needed
New-Item -ItemType Directory -Path $PluginDir -Force | Out-Null

# Copy plugin files
Write-Host "Installing plugin to $PluginDir..." -ForegroundColor Green
Copy-Item -Path "$PluginSrc\*" -Destination $PluginDir -Force

# Update base_url in __init__.py if ProxyUrl was provided
if ($ProxyUrl -ne "http://localhost:8787/v1") {
    Write-Host "Updating proxy URL to: $ProxyUrl" -ForegroundColor Yellow
    $InitFile = Join-Path $PluginDir "__init__.py"
    (Get-Content $InitFile) -replace "http://localhost:8787/v1", $ProxyUrl | Set-Content $InitFile
    (Get-Content $InitFile) -replace "http://localhost:8787/v1/models", "$ProxyUrl/models" | Set-Content $InitFile
}

Write-Host ""
Write-Host "Plugin installed successfully!" -ForegroundColor Green
Write-Host ""
Write-Host "To activate, add to your Hermes profile config.yaml:" -ForegroundColor Cyan
Write-Host "  model:" -ForegroundColor White
Write-Host "    default: deepseek/deepseek-v4-flash-0731" -ForegroundColor White
Write-Host "    provider: openrouter-better-proxy" -ForegroundColor White
Write-Host ""
Write-Host "Or set environment variable: $env:ORBP_PROXY_URL=$ProxyUrl" -ForegroundColor Cyan
