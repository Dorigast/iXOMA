# iXOMA

```markdown
# 🚀 WEEX Futures Trading Bot

**WEEX Exchange API Testing Solution (Hackathon Edition).**

A fully automated futures trading bot written in Python. It utilizes an SMA Crossover strategy and supports full remote control via Telegram. The bot is designed with a focus on safety (Safety Fuse) and API stability.

---

## 🌟 Key Features

### 🤖 Trading Logic
* **Strategy:** Simple Moving Average (SMA) Crossover + Volatility Filter.
* **Risk Management:** Hard-coded position sizes to comply with margin requirements (avoids `MIN_NOTIONAL` errors).
* **Force Entry:** "Quick Start" mode for hackathon judges — opens a trade immediately upon startup if no positions exist (to verify API functionality).

### 📱 Telegram Control
The bot doesn't just send notifications; it accepts commands:
* **Start/Stop:** Safely pause trading without killing the script.
* **Balance:** View Equity, Free Margin, and PnL in real-time.
* **Status:** Detailed report for every position (Entry Price, Current Price, TP/SL).
* **Panic Button:** Emergency close for all positions with one command.

---

## 🛠 Installation & Usage

### 1. Requirements
* Python 3.10+
* Libraries: `requests`, `python-dotenv`

### 2. Installation
```bash
# Clone the repository
git clone [https://github.com/your-username/weex-bot.git](https://github.com/your-username/weex-bot.git)
cd weex-bot

# Install dependencies
pip install requests python-dotenv

```

### 3. Configuration (.env)

Rename `.env.example` to `.env` and fill in your credentials:

```ini
WEEX_API_KEY=your_api_key
WEEX_API_SECRET=your_api_secret
WEEX_PASSPHRASE=your_passphrase

TELEGRAM_BOT_TOKEN=your_telegram_bot_token
TELEGRAM_CHAT_ID=your_telegram_chat_id

# Hackathon Settings:
FORCE_ENTRY=true     # Open trade immediately on start (for testing)
WEEX_DRY_RUN=false   # false = real trading

```

### 4. Run

```bash
python main.py --mode run-bot

```

---

## 🎮 Telegram Commands

Upon launch, the bot enters **PAUSED** mode. Send `/start_bot` to begin trading.

| Command | Description |
| --- | --- |
| `/start_bot` | **Start.** The bot begins searching for signals and trading. |
| `/stop` | **Pause.** The bot stops opening new trades. |
| `/status` | Report on open positions (Entry, Last, PnL, TP, SL). |
| `/balance` | Wallet report: Equity, Margin, Available. |
| `/close` | Manual position closing menu. |
| `/close all` | Close ALL positions at market price (requires confirmation). |
| `/help` | List of commands. |

---

## ⚙️ Position Sizing

To avoid API errors (`INVALID_ARGUMENT: stepSize requirement`), order sizes are hard-coded in `config.py`:

* **BTC:** 0.0002 (~18 USDT)
* **ETH:** 0.01 (~30 USDT)
* **SOL:** 0.1 (~20 USDT)
* **BNB:** 0.1 (~85 USDT, exchange min lot)
* **XRP:** 10 (~25 USDT)

---

## 📂 Project Structure

* `main.py` — Entry point. Main Event Loop and Telegram processing.
* `weex_api.py` — Wrapper for WEEX REST API (HMAC SHA256 signature).
* `config.py` — Configuration and environment variables.
* `strategy/` — Decision-making algorithms.
* `utils/` — Logger and Telegram notifier.

---

*Developed for WEEX API Testing.*

```

---

### Файл `requirements.txt`

Создайте файл с именем `requirements.txt` и вставьте туда эти строки (это список библиотек для установки):

```text
requests
python-dotenv

```
