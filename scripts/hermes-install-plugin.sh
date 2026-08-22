#!/bin/bash
# hermes-install-plugin.sh
# Install the openrouter-better-proxy Hermes plugin on Linux/macOS
#
# Usage:
#   ./hermes-install-plugin.sh [PROXY_URL]
#
# Arguments:
#   PROXY_URL  Optional. URL of the proxy (default: http://localhost:8787/v1)
#
# Example:
#   ./hermes-install-plugin.sh http://192.168.1.251:8787/v1

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PLUGIN_SRC="$SCRIPT_DIR/hermes-plugin/openrouter-better-proxy"
PROXY_URL="${1:-http://localhost:8787/v1}"

# Check if Hermes is installed
if [ ! -d "$HOME/.hermes" ]; then
    echo "ERROR: Hermes not found at $HOME/.hermes"
    echo "Please install Hermes first: https://hermes-agent.nousresearch.com/docs"
    exit 1
fi

# Create plugin directory if needed
PLUGIN_DIR="$HOME/.hermes/plugins/model-providers/openrouter-better-proxy"
mkdir -p "$PLUGIN_DIR"

# Copy plugin files
echo "Installing plugin to $PLUGIN_DIR..."
cp "$PLUGIN_SRC/__init__.py" "$PLUGIN_DIR/__init__.py"
cp "$PLUGIN_SRC/plugin.yaml" "$PLUGIN_DIR/plugin.yaml"

# Update base_url in __init__.py if PROXY_URL was provided
if [ "$PROXY_URL" != "http://localhost:8787/v1" ]; then
    echo "Updating proxy URL to: $PROXY_URL"
    sed -i "s|http://localhost:8787/v1|$PROXY_URL|g" "$PLUGIN_DIR/__init__.py"
    sed -i "s|http://localhost:8787/v1/models|$PROXY_URL/models|g" "$PLUGIN_DIR/__init__.py"
fi

echo "Plugin installed successfully!"
echo ""
echo "To activate, add to your Hermes profile config.yaml:"
echo "  model:"
echo "    default: deepseek/deepseek-v4-flash-0731"
echo "    provider: openrouter-better-proxy"
echo ""
echo "Or set environment variable: export ORBP_PROXY_URL=$PROXY_URL"
