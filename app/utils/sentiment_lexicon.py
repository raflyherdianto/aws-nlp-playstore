"""
Indonesian text-based sentiment scoring using InSet (Indonesia Sentiment Lexicon).

Loads positive.tsv and negative.tsv from app/data/ directory.
Handles negation, intensifiers, and bigram overrides.
Binary classification only: Positif / Negatif (no Netral).
"""

import math
import os
import csv
import logging

logger = logging.getLogger(__name__)

# ── Locate data directory (works both locally and inside Docker) ─────────────
_DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data')


def _load_inset_lexicon():
    """Load InSet positive + negative TSV files into a single dict: word → score."""
    lexicon = {}
    for filename in ('inset_negative.tsv', 'inset_positive.tsv'):
        filepath = os.path.join(_DATA_DIR, filename)
        if not os.path.exists(filepath):
            logger.warning(f'InSet file not found: {filepath}')
            continue
        with open(filepath, 'r', encoding='utf-8') as f:
            reader = csv.reader(f, delimiter='\t')
            header = next(reader, None)  # skip "word\tweight" header
            for row in reader:
                if len(row) < 2:
                    continue
                word = row[0].strip().lower()
                try:
                    weight = int(row[1].strip())
                except (ValueError, IndexError):
                    continue
                # Clean parenthetical annotations like "tersentuh (perasaan)"
                if '(' in word:
                    word = word.split('(')[0].strip()
                if not word:
                    continue
                lexicon[word] = weight
    logger.info(f'InSet lexicon loaded: {len(lexicon)} entries')
    return lexicon


# ── Load lexicon at module level (once) ──────────────────────────────────────
SENTIMENT_LEXICON = _load_inset_lexicon()

# ── App-review specific supplements (not in InSet) ──────────────────────────
_APP_REVIEW_SUPPLEMENTS = {
    # Performance issues
    'crash': -5, 'error': -5, 'bug': -5, 'lag': -5, 'ngelag': -5,
    'hang': -4, 'stuck': -4, 'loading': -2, 'buffering': -3,
    'force close': -5, 'fc': -4, 'ngehang': -5, 'ngadat': -4,
    # App actions
    'uninstall': -4, 'update': 1,
    # English common
    'good': 4, 'great': 4, 'nice': 3, 'awesome': 5, 'amazing': 5,
    'bad': -4, 'worst': -5, 'terrible': -5, 'sucks': -5, 'useless': -5,
    'love': 5, 'hate': -5, 'perfect': 5, 'excellent': 5,
    'slow': -4, 'fast': 3, 'broken': -5, 'smooth': 4,
    'annoying': -4, 'boring': -3, 'cool': 3, 'best': 5,
    # Indonesian slang sentiment
    'mantul': 5, 'josss': 5, 'joss': 5, 'kece': 5, 'gokil': 4,
    'ajib': 4, 'tokcer': 4, 'yahud': 4, 'juara': 4,
    'top': 5, 'nais': 4,
    'lemot': -5, 'lelet': -4,
}

# Merge supplements (don't overwrite InSet entries)
for _w, _s in _APP_REVIEW_SUPPLEMENTS.items():
    if _w not in SENTIMENT_LEXICON:
        SENTIMENT_LEXICON[_w] = _s


# ── Negators: flip the sign of the following sentiment word ──────────────────
NEGATORS = frozenset({
    'tidak', 'tak', 'bukan', 'belum', 'tanpa', 'jangan',
    'nggak', 'gak', 'ga', 'ngga', 'enggak', 'kagak',
    'ndak', 'gk', 'tdk', 'blm', 'nda', 'engga', 'kaga',
    'never', 'no', 'non',
})

# ── Intensifiers: multiply the score of the next/previous sentiment word ─────
INTENSIFIERS = {
    'sangat': 1.5, 'amat': 1.5, 'sekali': 1.5,
    'banget': 1.5, 'bgt': 1.5, 'sungguh': 1.5,
    'terlalu': 1.3, 'paling': 1.5, 'super': 1.5,
    'begitu': 1.3, 'really': 1.5, 'very': 1.5,
    'so': 1.3, 'nian': 1.4,
}

# ── Bigram overrides (checked before individual tokens) ──────────────────────
BIGRAM_SCORES = {
    'luar biasa': 5,
    'sangat bagus': 5,
    'sangat buruk': -5,
    'tidak bagus': -3,
    'force close': -5,
    'terima kasih': 4,
    'sia sia': -4,
    'kurang bagus': -2,
    'kurang baik': -2,
    'sangat membantu': 5,
    'sangat kecewa': -5,
    'tidak bisa': -3,
    'tidak berfungsi': -4,
    'sangat puas': 5,
    'sangat suka': 5,
    'sangat lambat': -5,
    'sangat lemot': -5,
    'mantap sekali': 5,
    'bagus sekali': 5,
    'jelek sekali': -5,
    'buruk sekali': -5,
    'sangat recommended': 5,
    'sangat mudah': 5,
    'sangat cepat': 5,
    'tidak responsif': -4,
    'tidak stabil': -4,
    'kurang puas': -3,
    'kurang responsif': -3,
}


def compute_sentiment_score(text):
    """
    Compute compound sentiment score for Indonesian text.

    The text should be basic-cleaned (lowercased, URLs removed) but NOT
    stopword-removed, so negators and intensifiers are preserved.

    Returns a float in the range [-1, 1].
    """
    if not text or not text.strip():
        return 0.0

    tokens = text.split()
    n = len(tokens)
    total_score = 0.0
    used = set()

    # ── Pass 1: Check bigrams ────────────────────────────────────────────
    for i in range(n - 1):
        bigram = f"{tokens[i]} {tokens[i + 1]}"
        if bigram in BIGRAM_SCORES:
            total_score += BIGRAM_SCORES[bigram]
            used.update({i, i + 1})

    # ── Pass 2: Multi-word InSet entries (2-3 word phrases) ──────────────
    for i in range(n):
        if i in used:
            continue
        # Check trigram
        if i + 2 < n:
            trigram = f"{tokens[i]} {tokens[i+1]} {tokens[i+2]}"
            if trigram in SENTIMENT_LEXICON:
                total_score += float(SENTIMENT_LEXICON[trigram])
                used.update({i, i+1, i+2})
                continue
        # Check bigram (InSet has multi-word entries)
        if i + 1 < n and i + 1 not in used:
            bigram = f"{tokens[i]} {tokens[i+1]}"
            if bigram in SENTIMENT_LEXICON:
                total_score += float(SENTIMENT_LEXICON[bigram])
                used.update({i, i+1})
                continue

    # ── Pass 3: Score individual sentiment words with context ────────────
    for i in range(n):
        if i in used:
            continue

        token = tokens[i]
        if token not in SENTIMENT_LEXICON:
            continue

        word_score = float(SENTIMENT_LEXICON[token])
        negate = False
        intensify = 1.0

        # Look back up to 2 tokens for negators / intensifiers
        for j in range(max(0, i - 2), i):
            if j in used:
                continue
            prev = tokens[j]
            if prev in NEGATORS:
                negate = True
                used.add(j)
            elif prev in INTENSIFIERS:
                intensify = max(intensify, INTENSIFIERS[prev])
                used.add(j)

        # Look ahead 1 token for post-modifier intensifier
        if i + 1 < n and (i + 1) not in used:
            nxt = tokens[i + 1]
            if nxt in INTENSIFIERS:
                intensify = max(intensify, INTENSIFIERS[nxt])
                used.add(i + 1)

        word_score *= intensify
        if negate:
            word_score *= -1.0

        total_score += word_score
        used.add(i)

    # ── VADER-style normalization to [-1, 1] ─────────────────────────────
    if total_score == 0.0:
        return 0.0
    alpha = 15.0
    return total_score / math.sqrt(total_score ** 2 + alpha)


def label_from_score(score):
    """Map compound score to binary sentiment label (no Netral)."""
    if score > 0:
        return 'Positif'
    return 'Negatif'


def apply_sentiment_labels(df):
    """
    Rating-based labeling for clean binary sentiment.

    - Rating 1-2 → Negatif  (strong signal)
    - Rating 4-5 → Positif  (strong signal)
    - Rating 3   → DROPPED  (ambiguous, hurts model accuracy)

    Clean labels from star ratings are the single biggest factor for
    achieving >90% accuracy on text-based classification.
    """
    from app.utils.preprocessing import clean_text

    df = df.copy()

    # Compute lexicon score (kept for analysis/display, not for labeling)
    df['sentiment_score'] = df['content'].apply(
        lambda x: compute_sentiment_score(clean_text(x) if isinstance(x, str) else '')
    )

    # Label from star rating only
    def _rating_label(rating):
        try:
            rating = int(rating)
        except (ValueError, TypeError):
            return None
        if rating >= 4:
            return 'Positif'
        elif rating <= 2:
            return 'Negatif'
        return None  # Rating 3 = ambiguous → drop

    df['sentiment'] = df['score'].apply(_rating_label)
    # Drop ambiguous rating-3 reviews
    df = df.dropna(subset=['sentiment'])
    df = df.reset_index(drop=True)
    return df
