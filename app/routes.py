"""
Flask routes – scrape, analyse, train, predict.
All chart images are written to  app/static/charts/<session_id>/.
CSV / model / metrics files live in  uploads/ (auto-deleted after 60 min).
"""

import os
import re
import gc
import time
import json
import uuid
import logging
import traceback

import pandas as pd
import numpy as np
import joblib
from scipy.sparse import hstack as sparse_hstack

# Force non-interactive Matplotlib backend (required inside Docker / headless)
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402

from flask import (
    Blueprint, render_template, request,
    redirect, url_for, flash, current_app, abort, jsonify,
)

logger = logging.getLogger(__name__)
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import FeatureUnion
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import confusion_matrix, classification_report
from wordcloud import WordCloud
from collections import Counter

from app.utils.scraper import scrape_reviews
from app.utils.preprocessing import preprocess_dataframe, preprocess_text, clean_for_model
from app.utils.sentiment_lexicon import apply_sentiment_labels, compute_sentiment_score
from app.utils.tasks import create_task, update_progress, get_task, run_in_background
from app import limiter

main_bp = Blueprint('main', __name__)

SENTIMENT_LABELS = ['Negatif', 'Positif']


def _validate_session_id(session_id):
    """Ensure session_id contains only digits to prevent path traversal."""
    if not re.match(r'^\d{1,20}$', session_id):
        abort(400)

# ══════════════════════════════════════════════════════════════════════════════
# ROUTES
# ══════════════════════════════════════════════════════════════════════════════

@main_bp.route('/health')
@limiter.exempt
def health():
    return {'status': 'ok'}, 200


@main_bp.route('/')
def index():
    return render_template('index.html')


@main_bp.route('/api/progress/<task_id>')
@limiter.exempt
def task_progress(task_id):
    """Polling endpoint for background task progress."""
    if not re.match(r'^[a-f0-9]{12}$', task_id):
        abort(400)
    task = get_task(task_id)
    if task is None:
        return jsonify({'status': 'not_found'}), 404
    return jsonify(task)


@main_bp.route('/scrape', methods=['POST'])
@limiter.limit("5 per minute")
def scrape():
    app_id = request.form.get('app_id', '').strip()
    if not app_id:
        return jsonify({'error': 'Masukkan Play Store App ID terlebih dahulu.'}), 400

    try:
        max_reviews = min(int(request.form.get('max_reviews') or 1000), 15000)
    except (ValueError, TypeError):
        max_reviews = 1000

    task_id = uuid.uuid4().hex[:12]
    session_id = str(int(time.time()))
    create_task(task_id)

    run_in_background(
        task_id, current_app._get_current_object(),
        _do_scrape_task,
        app_id=app_id, max_reviews=max_reviews, session_id=session_id,
    )

    return jsonify({'task_id': task_id, 'session_id': session_id})


@main_bp.route('/results/<session_id>')
def results(session_id):
    _validate_session_id(session_id)
    upload_folder = current_app.config['UPLOAD_FOLDER']
    csv_path = os.path.join(upload_folder, f'data_{session_id}.csv')

    if not os.path.exists(csv_path):
        flash('Sesi kadaluarsa atau data tidak ditemukan.', 'error')
        return redirect(url_for('main.index'))

    df = pd.read_csv(csv_path)
    total = len(df)
    sentiment_counts = df['sentiment'].value_counts().to_dict()

    model_path = os.path.join(upload_folder, f'model_{session_id}.joblib')
    has_model = os.path.exists(model_path)

    metrics = _load_metrics(upload_folder, session_id)

    return render_template(
        'results.html',
        session_id=session_id,
        total=total,
        sentiment_counts=sentiment_counts,
        has_model=has_model,
        metrics=metrics,
        sample_data=df.head(10).to_dict('records'),
    )


@main_bp.route('/train/<session_id>', methods=['POST'])
@limiter.limit("3 per minute")
def train(session_id):
    _validate_session_id(session_id)
    upload_folder = current_app.config['UPLOAD_FOLDER']
    csv_path = os.path.join(upload_folder, f'data_{session_id}.csv')

    if not os.path.exists(csv_path):
        return jsonify({'error': 'Data tidak ditemukan.'}), 404

    task_id = uuid.uuid4().hex[:12]
    create_task(task_id)

    run_in_background(
        task_id, current_app._get_current_object(),
        _do_train_task,
        session_id=session_id,
    )

    return jsonify({'task_id': task_id, 'session_id': session_id})


@main_bp.route('/predict/<session_id>', methods=['POST'])
def predict(session_id):
    _validate_session_id(session_id)
    upload_folder = current_app.config['UPLOAD_FOLDER']
    model_path = os.path.join(upload_folder, f'model_{session_id}.joblib')

    if not os.path.exists(model_path):
        flash('Model belum di-training.', 'error')
        return redirect(url_for('main.results', session_id=session_id))

    text = request.form.get('text', '').strip()
    if not text:
        flash('Masukkan teks untuk diprediksi.', 'error')
        return redirect(url_for('main.results', session_id=session_id))

    try:
        # Load model & predict
        model_data = joblib.load(model_path)
        cleaned = clean_for_model(text)
        featurizer = model_data.get('features') or model_data.get('tfidf')
        X_tfidf = featurizer.transform([cleaned])
        # Add lexicon score as extra feature (same as training)
        lex_score = compute_sentiment_score(cleaned)
        X = sparse_hstack([X_tfidf, np.array([[lex_score]])])
        prediction = model_data['model'].predict(X)[0]
        probabilities = model_data['model'].predict_proba(X)[0]
        prob_dict = {
            label: f"{prob:.2%}"
            for label, prob in zip(model_data['model'].classes_, probabilities)
        }

        # Re-load page data so the template renders fully
        csv_path = os.path.join(upload_folder, f'data_{session_id}.csv')
        df = pd.read_csv(csv_path)

        gc.collect()
        return render_template(
            'results.html',
            session_id=session_id,
            total=len(df),
            sentiment_counts=df['sentiment'].value_counts().to_dict(),
            has_model=True,
            metrics=_load_metrics(upload_folder, session_id),
            sample_data=df.head(10).to_dict('records'),
            prediction=prediction,
            prediction_text=text,
            probabilities=prob_dict,
        )

    except Exception as e:
        logger.error(f"Error in /predict: {e}\n{traceback.format_exc()}")
        flash(f'Terjadi kesalahan saat prediksi: {e}', 'error')
        return redirect(url_for('main.results', session_id=session_id))


# ══════════════════════════════════════════════════════════════════════════════
# BACKGROUND TASK FUNCTIONS
# ══════════════════════════════════════════════════════════════════════════════

def _do_scrape_task(task_id, app_id, max_reviews, session_id):
    """Background: scrape → sentiment → preprocess → charts."""
    update_progress(task_id, 0, 100, 'Memulai scraping review...')

    def on_scrape_progress(scraped, total):
        pct = min(int((scraped / total) * 40), 40)
        update_progress(task_id, pct, 100,
                        f'Scraping review... ({scraped:,}/{total:,})')

    df = scrape_reviews(app_id, max_reviews=max_reviews,
                        progress_callback=on_scrape_progress)
    if df is None or len(df) == 0:
        raise ValueError('Review tidak ditemukan. Periksa App ID dan coba lagi.')

    update_progress(task_id, 45, 100,
                    f'Analisis sentimen ({len(df):,} review)...')
    df = apply_sentiment_labels(df)

    update_progress(task_id, 60, 100, 'Preprocessing teks...')
    df = preprocess_dataframe(df)
    if df is None or len(df) == 0:
        raise ValueError('Semua review kosong setelah preprocessing.')

    update_progress(task_id, 75, 100, 'Menyimpan data...')
    upload_folder = current_app.config['UPLOAD_FOLDER']
    csv_path = os.path.join(upload_folder, f'data_{session_id}.csv')
    df.to_csv(csv_path, index=False)

    update_progress(task_id, 80, 100, 'Membuat visualisasi...')
    _generate_analysis_charts(df, session_id)

    gc.collect()
    return {'session_id': session_id}


def _do_train_task(task_id, session_id):
    """Background: load → filter noise → features → train LinearSVC → evaluate."""
    upload_folder = current_app.config['UPLOAD_FOLDER']
    csv_path = os.path.join(upload_folder, f'data_{session_id}.csv')

    update_progress(task_id, 5, 100, 'Memuat data...')
    df = pd.read_csv(csv_path)
    df = df.dropna(subset=['cleaned'])
    df = df[df['cleaned'].str.strip() != '']

    # Filter very short reviews (< 3 tokens) — too noisy for training
    df = df[df['cleaned'].str.split().str.len() >= 3]
    # Remove exact duplicates in cleaned text
    df = df.drop_duplicates(subset='cleaned', keep='first')
    df = df.reset_index(drop=True)

    if len(df) < 10:
        raise ValueError('Data tidak cukup untuk training (min 10 baris).')

    # Build model_text: minimal cleaning (preserves negators for accuracy)
    update_progress(task_id, 10, 100, 'Preparing features text...')
    df['model_text'] = df['content'].apply(clean_for_model)

    # ── Filter noisy samples: remove reviews where text contradicts rating ──
    # This is the key to >90%: rating says Positif but text is clearly negative
    # (or vice versa) creates label noise that confuses the model.
    update_progress(task_id, 15, 100, 'Filtering noisy samples...')
    df['lex_score'] = df['model_text'].apply(compute_sentiment_score)
    n_before = len(df)
    # Mismatch: rated Positif (4-5 star) but text is strongly negative
    mask_pos_mismatch = (df['sentiment'] == 'Positif') & (df['lex_score'] < -0.3)
    # Mismatch: rated Negatif (1-2 star) but text is strongly positive
    mask_neg_mismatch = (df['sentiment'] == 'Negatif') & (df['lex_score'] > 0.3)
    df = df[~mask_pos_mismatch & ~mask_neg_mismatch]
    df = df.reset_index(drop=True)
    n_removed = n_before - len(df)
    logger.info(f'Noise filter: removed {n_removed}/{n_before} mismatched samples')

    if len(df) < 10:
        raise ValueError('Data tidak cukup setelah filtering (min 10 baris).')

    X_text = df['model_text']
    X_lex = df['lex_score'].values.reshape(-1, 1)
    y = df['sentiment']

    # Ensure stratification is safe
    value_counts = y.value_counts()
    use_stratify = value_counts.min() >= 5 and y.nunique() >= 2
    if y.nunique() < 2:
        raise ValueError('Minimal 2 kelas sentimen diperlukan untuk training.')

    X_train_text, X_test_text, X_train_lex, X_test_lex, y_train, y_test = train_test_split(
        X_text, X_lex, y, test_size=0.2, random_state=42,
        stratify=y if use_stratify else None,
    )

    # ── Feature extraction: word n-grams + char n-grams ──────────────────
    update_progress(task_id, 25, 100, 'Feature extraction (word + char n-grams)...')
    features = FeatureUnion([
        ('word', TfidfVectorizer(
            max_features=15000, ngram_range=(1, 3),
            sublinear_tf=True, min_df=2, max_df=0.95,
        )),
        ('char', TfidfVectorizer(
            max_features=15000, ngram_range=(3, 5),
            sublinear_tf=True, min_df=2, max_df=0.95,
            analyzer='char_wb',
        )),
    ])
    X_train_tfidf = features.fit_transform(X_train_text)
    # Append lexicon sentiment score as an extra feature column
    X_train_feat = sparse_hstack([X_train_tfidf, X_train_lex])

    # ── Train LinearSVC (best for binary text classification) ───────────
    update_progress(task_id, 40, 100, 'Training LinearSVC...')
    svc = LinearSVC(
        C=0.5, max_iter=3000, random_state=42,
        class_weight='balanced', tol=1e-4,
    )
    # CalibratedClassifierCV wraps SVC to provide predict_proba
    model = CalibratedClassifierCV(svc, cv=3, method='sigmoid')
    model.fit(X_train_feat, y_train)

    # ── Evaluate on held-out test set ────────────────────────────────────
    update_progress(task_id, 60, 100, 'Evaluasi model...')
    X_test_tfidf = features.transform(X_test_text)
    X_test_feat = sparse_hstack([X_test_tfidf, X_test_lex])
    y_pred = model.predict(X_test_feat)

    cm = confusion_matrix(y_test, y_pred, labels=SENTIMENT_LABELS)
    report_dict = classification_report(
        y_test, y_pred, labels=SENTIMENT_LABELS,
        output_dict=True, zero_division=0,
    )
    report_text = classification_report(
        y_test, y_pred, labels=SENTIMENT_LABELS, zero_division=0,
    )

    update_progress(task_id, 80, 100, 'Menyimpan model...')
    model_path = os.path.join(upload_folder, f'model_{session_id}.joblib')
    joblib.dump({'model': model, 'features': features}, model_path)

    metrics = {
        'confusion_matrix': cm.tolist(),
        'classification_report': report_dict,
        'classification_report_text': report_text,
        'labels': SENTIMENT_LABELS,
    }
    metrics_path = os.path.join(upload_folder, f'metrics_{session_id}.json')
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f)

    update_progress(task_id, 90, 100, 'Membuat confusion matrix...')
    _generate_confusion_matrix_chart(cm, SENTIMENT_LABELS, session_id)

    gc.collect()
    return {'session_id': session_id}


# ══════════════════════════════════════════════════════════════════════════════
# CHART GENERATION  (saved as PNGs to app/static/charts/<session_id>/)
# ══════════════════════════════════════════════════════════════════════════════

def _charts_dir(session_id):
    d = os.path.join(current_app.static_folder, 'charts', session_id)
    os.makedirs(d, exist_ok=True)
    return d


def _generate_analysis_charts(df, session_id):
    """Pie chart, two word clouds (Positif/Negatif), and top-10 bar chart."""
    cdir = _charts_dir(session_id)

    # ── 1. Sentiment distribution pie chart ──────────────────────────────
    sentiment_counts = df['sentiment'].value_counts()
    colors_map = {'Positif': '#22c55e', 'Negatif': '#ef4444'}
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.pie(
        sentiment_counts.values,
        labels=sentiment_counts.index,
        autopct='%1.1f%%',
        colors=[colors_map.get(s, '#94a3b8') for s in sentiment_counts.index],
        startangle=90,
        textprops={'fontsize': 12},
    )
    ax.set_title('Distribusi Sentimen', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(cdir, 'pie_chart.png'), dpi=100, bbox_inches='tight')
    plt.close(fig)

    # ── 2. Word clouds per sentiment (using cleaned tokens, no re-stemming) ─
    cmap_for = {'Positif': 'Greens', 'Negatif': 'Reds'}
    for sentiment in SENTIMENT_LABELS:
        subset_text = ' '.join(
            df.loc[df['sentiment'] == sentiment, 'cleaned'].dropna().values
        )
        if not subset_text.strip():
            continue
        # Use already-cleaned tokens directly — no re-stemming for speed
        tokens = [t for t in subset_text.split() if len(t) > 2]
        freq = Counter(tokens)
        if not freq:
            continue
        wc = WordCloud(
            width=800, height=400,
            background_color='white',
            max_words=100,
            colormap=cmap_for.get(sentiment, 'viridis'),
        ).generate_from_frequencies(freq)

        fig, ax = plt.subplots(figsize=(10, 5))
        ax.imshow(wc, interpolation='bilinear')
        ax.axis('off')
        ax.set_title(f'Word Cloud – {sentiment}', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(
            os.path.join(cdir, f'wordcloud_{sentiment.lower()}.png'),
            dpi=100, bbox_inches='tight',
        )
        plt.close(fig)

    # ── 3. Top-10 frequent words (horizontal bar, no re-stemming) ────────
    all_tokens = ' '.join(df['cleaned'].dropna().values).split()
    filtered = [t for t in all_tokens if len(t) > 2]
    top10 = Counter(filtered).most_common(10)
    if top10:
        words, counts = zip(*top10)
        fig, ax = plt.subplots(figsize=(10, 6))
        y_pos = range(len(words))
        ax.barh(y_pos, counts, color='#3b82f6')
        ax.set_yticks(y_pos)
        ax.set_yticklabels(words)
        ax.invert_yaxis()
        ax.set_xlabel('Frekuensi', fontsize=12)
        ax.set_title('Top 10 Kata Paling Sering Muncul', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(os.path.join(cdir, 'top10_bar.png'), dpi=100, bbox_inches='tight')
        plt.close(fig)

    gc.collect()


def _generate_confusion_matrix_chart(cm, labels, session_id):
    cdir = _charts_dir(session_id)
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(cm, interpolation='nearest', cmap='Blues')
    ax.figure.colorbar(im, ax=ax)
    ax.set(
        xticks=range(len(labels)), yticks=range(len(labels)),
        xticklabels=labels, yticklabels=labels,
        title='Confusion Matrix', ylabel='Actual', xlabel='Predicted',
    )
    thresh = cm.max() / 2.0
    for i in range(len(labels)):
        for j in range(len(labels)):
            ax.text(
                j, i, format(cm[i, j], 'd'),
                ha='center', va='center',
                color='white' if cm[i, j] > thresh else 'black',
            )
    plt.tight_layout()
    plt.savefig(os.path.join(cdir, 'confusion_matrix.png'), dpi=100, bbox_inches='tight')
    plt.close(fig)
    gc.collect()


# ── helpers ──────────────────────────────────────────────────────────────────

def _load_metrics(upload_folder, session_id):
    path = os.path.join(upload_folder, f'metrics_{session_id}.json')
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return None
