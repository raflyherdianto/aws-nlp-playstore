import nltk
import pytest
from app import create_app

# Ensure NLTK tokenizer data is available for tests
nltk.download('punkt_tab', quiet=True)


@pytest.fixture
def app():
    """Create application for testing."""
    app = create_app()
    app.config.update({
        'TESTING': True,
        'WTF_CSRF_ENABLED': False,  # Disable CSRF for testing
    })
    yield app


@pytest.fixture
def client(app):
    """Flask test client."""
    return app.test_client()
