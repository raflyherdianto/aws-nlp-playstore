# NLP Sentiment Analyzer — Google Play Store Reviews

![Python](https://img.shields.io/badge/Python-3.12-blue?logo=python&logoColor=white)
![Flask](https://img.shields.io/badge/Flask-3.0-green?logo=flask&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.5-orange?logo=scikit-learn&logoColor=white)
![Docker](https://img.shields.io/badge/Docker-Multi--Arch-2496ED?logo=docker&logoColor=white)
![AWS](https://img.shields.io/badge/AWS-EC2_Spot-FF9900?logo=amazonec2&logoColor=white)
![Tailwind CSS](https://img.shields.io/badge/Tailwind_CSS-3.0-38B2AC?logo=tailwindcss&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-yellow)

An end-to-end **Indonesian NLP Sentiment Analysis** web application that scrapes Google Play Store reviews, classifies sentiment using a LinearSVC model, and provides interactive visualizations — deployed on **AWS EC2 Spot Instance** with automated CI/CD.

> **Live Demo:** [https://nlp1.bangraply.cloud](https://nlp1.bangraply.cloud)

---

## Screenshots

| Landing Page | Result Page |
|:---:|:---:|
| ![Landing Page](Landing%20Page.png) | ![Result Page](Result%20Page.png) |

---

## Project Overview

This application performs binary sentiment classification (Positif / Negatif) on Indonesian app reviews scraped from the Google Play Store. It combines traditional NLP preprocessing with machine learning to achieve **>90% accuracy** on real-world data.

### Key Features

- **Play Store Scraper** — Scrapes up to 15,000 Indonesian reviews via `google-play-scraper` with real-time progress tracking
- **Sentiment Analysis** — Rating-based labeling with InSet lexicon scoring for noise filtering
- **Machine Learning** — LinearSVC + TF-IDF (word + character n-grams) with calibrated probability output
- **Interactive Visualizations** — Pie chart, per-sentiment word clouds, and top-10 frequent words bar chart
- **Single-text Prediction** — Test the trained model on any custom text input
- **Background Processing** — Async scraping & training via threading with live progress bars
- **Auto-cleanup** — APScheduler deletes temporary files after 15 minutes
- **Production Ready** — Gunicorn WSGI, CSRF protection, rate limiting, Docker multi-stage build
- **Multi-arch Docker** — GHCR-hosted images for both `amd64` and `arm64` (AWS Graviton)
- **Spot Instance Resilient** — Auto-restart via systemd on instance recovery

---

## Training Methodology

### Labeling Strategy

Instead of relying solely on lexicon-based scoring (which is noisy), we use a **rating-based labeling** approach:

| Star Rating | Sentiment Label |
|:-----------:|:---------------:|
| 1–2 ★       | **Negatif**     |
| 3 ★         | *Dropped* (ambiguous) |
| 4–5 ★       | **Positif**     |

Ambiguous 3-star reviews are excluded from training data, and reviews where the text strongly contradicts the rating (lexicon score threshold ±0.3) are filtered out as noise.

### Text Preprocessing

1. **Lowercasing** + URL/special character removal
2. **Slang normalization** — 700+ Indonesian slang mappings (e.g., "gak" → "tidak", "bgt" → "banget")
3. **Minimal stop word removal** for word clouds only — the ML model receives full text to preserve negation patterns ("tidak bagus" stays intact)

### Feature Engineering

| Feature Type | Config | Description |
|:--|:--|:--|
| Word n-grams | TF-IDF, (1,3), 15K features | Captures phrases like "tidak bagus" |
| Char n-grams | TF-IDF, (3,5), 15K features, `char_wb` | Captures Indonesian morphology (prefixes: *me-*, *ber-*, *di-*) |
| Lexicon score | InSet compound score | Pre-computed sentiment signal from 7,000+ word lexicon |

### Model

- **LinearSVC** with `class_weight='balanced'` wrapped in `CalibratedClassifierCV` (sigmoid, 3-fold) for probability estimation
- Regularization: `C=0.5`, convergence tolerance: `1e-4`
- Achieves **~95% accuracy** on held-out test set

### NLP Resources

This project uses the following open-source Indonesian NLP resources:

- **[InSet (Indonesia Sentiment Lexicon)](https://github.com/fajri91/InSet)** by [Fajri Koto](https://github.com/fajri91) — 7,000+ sentiment-scored Indonesian words (`positive.tsv` and `negative.tsv`), used for lexicon-based sentiment scoring and noise filtering
- **[NLP Bahasa Resources](https://github.com/louisowen6/NLP_bahasa_resources)** by [Louis Owen](https://github.com/louisowen6) — Combined slang words dictionary, stop words list, and root words dictionary for Indonesian text preprocessing

---

## Project Structure

```
nlp-aws/
├── app/
│   ├── data/                    # NLP resource files
│   │   ├── inset_negative.tsv   # InSet negative lexicon (─5 to ─1)
│   │   ├── inset_positive.tsv   # InSet positive lexicon (+1 to +5)
│   │   ├── slang_words.json     # 700+ slang → normalized mappings
│   │   ├── stop_words.txt       # Indonesian stop words
│   │   └── root_words.txt       # Indonesian root word dictionary
│   ├── static/
│   │   └── src/input.css        # Tailwind CSS source
│   ├── templates/
│   │   ├── base.html            # Layout with navbar & flash messages
│   │   ├── index.html           # Scrape form + progress bar
│   │   ├── results.html         # Charts, ML training, prediction
│   │   ├── 404.html             # Not found page
│   │   ├── 429.html             # Rate limit exceeded page
│   │   └── 500.html             # Server error page
│   ├── utils/
│   │   ├── preprocessing.py     # Text cleaning, slang norm, tokenization
│   │   ├── sentiment_lexicon.py # InSet lexicon loader + scoring engine
│   │   ├── scraper.py           # Google Play Store scraper
│   │   ├── scheduler.py         # APScheduler for auto-cleanup
│   │   └── tasks.py             # Background task manager (threading)
│   ├── __init__.py              # Flask app factory
│   └── routes.py                # All routes + ML pipeline + chart gen
├── tests/                       # Pytest test suite
├── .github/workflows/           # CI/CD pipeline
├── Dockerfile                   # Multi-stage: Node (Tailwind) → Python
├── docker-compose.yml           # Dev/local config (builds image locally)
├── docker-compose.prod.yml      # EC2 production reference (pulls from GHCR)
├── requirements.txt             # Python dependencies
├── tailwind.config.js           # Tailwind CSS configuration
├── main.py                      # Application entrypoint
└── .env                         # Environment variables (SECRET_KEY)
```

---

## Tech Stack

| Layer | Technology |
|:--|:--|
| **Backend** | Flask 3.0, Gunicorn (gthread) |
| **ML/NLP** | scikit-learn 1.5, NLTK, Sastrawi, InSet Lexicon |
| **Frontend** | Tailwind CSS 3, Jinja2, vanilla JS (AJAX + progress bars) |
| **Data** | Pandas, google-play-scraper |
| **Visualization** | Matplotlib, WordCloud |
| **Security** | Flask-WTF (CSRF), Flask-Limiter (rate limiting) |
| **Infrastructure** | Docker, Docker Compose, GHCR, AWS EC2 (Spot), Cloudflare Tunnel |
| **CI/CD** | GitHub Actions (test → build multi-arch → deploy) |

---

## Getting Started (Local with Docker)

### Prerequisites

- [Docker](https://docs.docker.com/get-docker/) and [Docker Compose](https://docs.docker.com/compose/install/) installed
- Git

### 1. Clone the repository

```bash
git clone https://github.com/raflyherdianto/aws-nlp-playstore.git
cd aws-nlp-playstore
```

### 2. Configure environment variables

```bash
cp .env.example .env
# Edit .env and set a SECRET_KEY:
# SECRET_KEY=your-random-secret-key-here
```

Or generate one automatically:

```bash
echo "SECRET_KEY=$(python3 -c 'import secrets; print(secrets.token_hex(32))')" > .env
```

### 3. Build and run

```bash
docker compose up --build -d
```

The application will be available at **http://localhost:5000**.

### 4. Verify it's running

```bash
curl http://localhost:5000/health
# Expected: {"status": "ok"}
```

### 5. Stop the application

```bash
docker compose down
```

---

## Usage

1. **Enter a Play Store App ID** (e.g., `com.gojek.app`, `com.tokopedia.tkpd`)
2. **Set the number of reviews** to scrape (100–15,000)
3. Click **"Mulai Scraping & Analisis"** — watch the progress bar
4. View the **sentiment distribution**, **word clouds**, and **top words**
5. Click **"Train Model"** to train the LinearSVC classifier
6. Use the **prediction form** to test sentiment on custom text

---

## Resource Configuration (AWS EC2 t4g.medium Spot Instance)

The application runs on `t4g.medium` (2 vCPU ARM Graviton, 4 GB RAM) as a **Persistent Spot Instance**:

| Resource | Value |
|:--|:--|
| Instance type | `t4g.medium` (ARM64 Graviton) |
| Request type | Spot — Persistent |
| Memory limit | 1,536 MB (container) |
| CPU limit | 1.5 cores |
| Workers | 2 (gthread with 2 threads each) |
| Timeout | 300 seconds |
| Log rotation | 10 MB × 3 files |
| Docker image | Multi-arch (amd64 + arm64) via GHCR |
| Auto-restart | systemd service on boot (Spot recovery) |

---

## Acknowledgements

- **[Fajri Koto](https://github.com/fajri91)** — [InSet: Indonesia Sentiment Lexicon](https://github.com/fajri91/InSet) for the comprehensive positive/negative sentiment word lists with weight scores
- **[Louis Owen](https://github.com/louisowen6)** — [NLP Bahasa Resources](https://github.com/louisowen6/NLP_bahasa_resources) for the combined slang words, stop words, and root words dictionaries for Indonesian NLP

---

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.
