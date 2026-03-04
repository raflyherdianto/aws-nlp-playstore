"""
Indonesian text preprocessing with external NLP resources.

Uses:
- combined_slang_words.json  → slang/abbreviation normalization
- combined_stop_words.txt    → Indonesian stop word removal
- combined_root_words.txt    → root word dictionary (optional stemming validation)
- Sastrawi stemmer           → Indonesian stemming (optional, resource-heavy)
- NLTK punkt                 → tokenization

Designed for memory efficiency on AWS Free Tier (t3.micro / 1 GB RAM).
"""

import os
import re
import gc
import json
import logging
import nltk

logger = logging.getLogger(__name__)

# ── Locate data directory ────────────────────────────────────────────────────
_DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data')

# ── NLTK data download (runs once) ──────────────────────────────────────────
_NLTK_DIR = os.environ.get('NLTK_DATA', '/root/nltk_data')
if _NLTK_DIR not in nltk.data.path:
    nltk.data.path.insert(0, _NLTK_DIR)

for _resource in ('punkt', 'punkt_tab'):
    try:
        nltk.data.find(f'tokenizers/{_resource}')
    except LookupError:
        try:
            nltk.download(_resource, download_dir=_NLTK_DIR, quiet=True)
            logger.info(f'Downloaded NLTK resource: {_resource}')
        except Exception as e:
            logger.warning(f'Failed to download NLTK resource {_resource}: {e}')

from nltk.tokenize import word_tokenize  # noqa: E402


# ══════════════════════════════════════════════════════════════════════════════
# SINGLETON RESOURCE LOADERS
# ══════════════════════════════════════════════════════════════════════════════

_slang_dict = None
_stop_words = None
_root_words = None
_stemmer = None


def _load_slang_dict():
    """Load slang → normalized word mapping from JSON file."""
    filepath = os.path.join(_DATA_DIR, 'slang_words.json')
    if not os.path.exists(filepath):
        logger.warning(f'Slang words file not found: {filepath}')
        return {}
    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)
    logger.info(f'Slang dictionary loaded: {len(data)} entries')
    return data


def _load_stop_words():
    """Load stop words from text file (one word per line)."""
    words = set()
    filepath = os.path.join(_DATA_DIR, 'stop_words.txt')
    if os.path.exists(filepath):
        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                w = line.strip().lower()
                if w:
                    words.add(w)
    # Add extra noise tokens common in app reviews
    words.update({
        'app', 'aplikasi', 'nya', 'dong', 'deh', 'sih', 'nih', 'lah',
        'kok', 'kan', 'kah', 'pun', 'lho', 'tuh', 'aja',
    })
    logger.info(f'Stop words loaded: {len(words)} entries')
    return words


def _load_root_words():
    """Load root words from text file (one word per line)."""
    words = set()
    filepath = os.path.join(_DATA_DIR, 'root_words.txt')
    if os.path.exists(filepath):
        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                w = line.strip().lower()
                if w:
                    words.add(w)
    logger.info(f'Root words loaded: {len(words)} entries')
    return words


def get_slang_dict():
    global _slang_dict
    if _slang_dict is None:
        _slang_dict = _load_slang_dict()
    return _slang_dict


def get_stop_words():
    global _stop_words
    if _stop_words is None:
        _stop_words = _load_stop_words()
    return _stop_words


def get_root_words():
    global _root_words
    if _root_words is None:
        _root_words = _load_root_words()
    return _root_words


def get_stemmer():
    """Lazy-load Sastrawi stemmer (expensive: ~200 MB, slow init)."""
    global _stemmer
    if _stemmer is None:
        from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
        _stemmer = StemmerFactory().create_stemmer()
    return _stemmer


# ══════════════════════════════════════════════════════════════════════════════
# TEXT PROCESSING FUNCTIONS
# ══════════════════════════════════════════════════════════════════════════════

def clean_text(text):
    """Lowercase, strip URLs, non-alpha characters, and collapse whitespace."""
    if not isinstance(text, str):
        return ''
    text = text.lower()
    text = re.sub(r'http\S+|www\S+|https\S+', '', text)   # URLs
    text = re.sub(r'[^a-z\s]', '', text)                    # keep only letters
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def normalize_slang(text):
    """Replace slang/abbreviations with normalized forms."""
    slang = get_slang_dict()
    if not slang:
        return text
    tokens = text.split()
    normalized = []
    for token in tokens:
        normalized.append(slang.get(token, token))
    return ' '.join(normalized)


def preprocess_text(text, use_stemming=False):
    """Clean → normalize slang → tokenize → remove stopwords → (optional) stem."""
    text = clean_text(text)
    text = normalize_slang(text)
    tokens = word_tokenize(text)
    stop_words = get_stop_words()
    tokens = [t for t in tokens if t not in stop_words and len(t) > 2]

    if use_stemming:
        stemmer = get_stemmer()
        tokens = [stemmer.stem(t) for t in tokens]

    return ' '.join(tokens)


def clean_for_model(text):
    """Minimal cleaning for ML features: lowercase + slang norm, NO stop word removal.

    Preserving all words (especially negators like 'tidak', 'bukan', 'ga')
    is critical for sentiment classification accuracy.
    """
    text = clean_text(text)
    text = normalize_slang(text)
    return text


def preprocess_dataframe(df):
    """Add a 'cleaned' column to *df* and drop empty rows."""
    df = df.copy()
    df['cleaned'] = df['content'].apply(lambda x: preprocess_text(x, use_stemming=False))
    df = df[df['cleaned'].str.strip() != '']
    df = df.reset_index(drop=True)
    gc.collect()
    return df
