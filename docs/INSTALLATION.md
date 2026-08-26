# Installation Guide

This guide covers deploying the OpenRouter Better Proxy on a Linux server (Debian/Ubuntu).

## Prerequisites

- Python 3.10+
- systemd (for service management)
- OpenRouter API key ([get one](https://openrouter.ai/keys))

## Step 1: Create Application Directory

```bash
sudo mkdir -p /opt/openrouter-better-proxy
sudo chown -R $USER:$USER /opt/openrouter-better-proxy
```

## Step 2: Clone or Copy Repository

### Option A: From GitHub (recommended for new deployments)

```bash
cd /opt
git clone https://github.com/rafpigna/openrouter-better-proxy.git openrouter-better-proxy
cd openrouter-better-proxy
```

### Option B: Copy from existing deployment

```bash
rsync -avz user@existing-server:/opt/openrouter-better-proxy/ /opt/openrouter-better-proxy/
```

## Step 3: Create Virtual Environment

```bash
cd /opt/openrouter-better-proxy
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Step 4: Configure Application

### 4.1 Copy Configuration

```bash
cp routing_config.yaml routing_config.yaml
```

Edit `routing_config.yaml` to match your needs (see [Configuration](#configuration)).

### 4.2 Create .env File

```bash
cp .env.example .env
```

Edit `.env` and add your OpenRouter API key:

```bash
OPENROUTER_API_KEY=sk-or-v1-your-key-here
```

## Step 5: Create Systemd Service

```bash
sudo cp openrouter-better-proxy.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable openrouter-better-proxy
sudo systemctl start openrouter-better-proxy
```

## Step 6: Verify Installation

```bash
# Check service status
sudo systemctl status openrouter-better-proxy --no-pager

# Check logs
sudo journalctl -u openrouter-better-proxy -f

# Test health endpoint
curl http://localhost:8787/health

# Test models endpoint
curl http://localhost:8787/v1/models

# Access dashboard (if enabled)
# Open http://localhost:8787/dashboard/ in your browser
```

## Configuration

### routing_config.yaml

```yaml
server:
  host: "0.0.0.0"      # Bind address (use 127.0.0.1 for localhost-only)
  port: 8787
  dashboard: true       # Enable web dashboard (default: true)

models:
  "deepseek/deepseek-v4-flash-0731":
    quantizations: ["fp8", "fp4"]      # Priority order (tiered)
    providers: ["deepseek", "deepinfra", "streamlake", "gmicloud"]
    max_price:                     # $/M token (dollars per million tokens)
      input: 0.10
      completion: 0.25
      cache: 0.05

migration:
  enabled: true
  hysteresis_mult: 3.0
  est_turns_per_session: 50
  r_cache_estimate: 300000
  out_per_turn_estimate: 40000

refresh:
  interval_minutes: 30
  price_change_threshold: 1        # Price-change threshold in % (10 = 10%)
  times:
    - "10:00"
    - "10:05"
    - "10:10"
    - "16:00"
    - {from: "18:00", to: "22:00", step: "05m"}

health:
  initial_cooldown_seconds: 300
  consecutive_threshold: 3
  escalation_seconds: [3600, 43200]
  max_cooldown_seconds: 43200
```

### .env

```bash
OPENROUTER_API_KEY=sk-or-v1-your-key-here
```

## Hermes Plugin Installation

### Automatic (recommended)

```bash
# Linux
./scripts/hermes-install-plugin.sh http://YOUR_SERVER:8787/v1

# Windows (PowerShell)
.\scripts\hermes-install-plugin.ps1 -ProxyUrl "http://YOUR_SERVER:8787/v1"
```

### Manual

1. Copy `hermes-plugin/openrouter-better-proxy/` to your Hermes profile:
   - Linux: `~/.hermes/plugins/model-providers/`
   - Windows: `%LOCALAPPDATA%\hermes\plugins\model-providers\`

2. Update `__init__.py` to point to your proxy URL.

3. Add to your Hermes profile `config.yaml`:

```yaml
model:
  default: deepseek/deepseek-v4-flash-0731
  provider: openrouter-better-proxy
```

## Troubleshooting

### Service won't start

```bash
# Check logs
sudo journalctl -u openrouter-better-proxy -n 50 --no-pager

# Test config
cd /opt/openrouter-better-proxy && source venv/bin/activate && python -c "from config import config; print(config)"
```

### Permission denied

```bash
sudo chown -R $USER:$USER /opt/openrouter-better-proxy
sudo chmod 755 /opt/openrouter-better-proxy
```

### Port already in use

Edit `routing_config.yaml` to use a different port, or stop the conflicting service.
