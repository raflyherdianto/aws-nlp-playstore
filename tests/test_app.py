"""Basic tests for the NLP Sentiment Analyzer application."""


def test_index_page(client):
    """GET / should return 200 with the scrape form."""
    response = client.get('/')
    assert response.status_code == 200
    assert b'Analisis Sentimen Play Store' in response.data


def test_health_endpoint(client):
    """GET /health should return 200 JSON."""
    response = client.get('/health')
    assert response.status_code == 200
    data = response.get_json()
    assert data['status'] == 'ok'


def test_scrape_requires_app_id(client):
    """POST /scrape without app_id should redirect back with error flash."""
    response = client.post('/scrape', data={'app_id': '', 'max_reviews': '100'})
    assert response.status_code == 302  # redirect


def test_results_invalid_session(client):
    """GET /results/<bad_session> should return 400 for path traversal attempt."""
    response = client.get('/results/../../etc/passwd')
    assert response.status_code in (400, 404)


def test_results_valid_session_missing_data(client):
    """GET /results/999 should redirect when CSV doesn't exist."""
    response = client.get('/results/999')
    assert response.status_code == 302  # flash + redirect


def test_404_page(client):
    """Non-existent route should return 404 with custom page."""
    response = client.get('/nonexistent-route')
    assert response.status_code == 404
    assert b'Halaman Tidak Ditemukan' in response.data


def test_preprocess_text():
    """preprocess_text should clean and tokenize Indonesian text."""
    from app.utils.preprocessing import preprocess_text

    result = preprocess_text('Aplikasi ini SANGAT bagus!! http://test.com 😀')
    # Should lowercase, remove URLs, non-alpha, and stopwords
    assert 'http' not in result
    assert result == result.lower()
    assert '!!' not in result


def test_sentiment_lexicon_positive():
    """Positive Indonesian text should get a positive sentiment score."""
    from app.utils.sentiment_lexicon import compute_sentiment_score, label_from_score

    score = compute_sentiment_score('aplikasi ini sangat bagus dan mantap')
    assert score > 0
    assert label_from_score(score) == 'Positif'


def test_sentiment_lexicon_negative():
    """Negative Indonesian text should get a negative sentiment score."""
    from app.utils.sentiment_lexicon import compute_sentiment_score, label_from_score

    score = compute_sentiment_score('aplikasi jelek lambat dan sering error')
    assert score < 0
    assert label_from_score(score) == 'Negatif'


def test_sentiment_lexicon_negation():
    """Negated positive word should flip sentiment."""
    from app.utils.sentiment_lexicon import compute_sentiment_score

    score_pos = compute_sentiment_score('bagus')
    score_neg = compute_sentiment_score('tidak bagus')
    assert score_pos > 0
    assert score_neg < 0
