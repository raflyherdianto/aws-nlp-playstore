import os
import logging
from flask import Flask, render_template
from dotenv import load_dotenv
from flask_wtf.csrf import CSRFProtect
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address

load_dotenv()

csrf = CSRFProtect()
limiter = Limiter(
    key_func=get_remote_address,
    default_limits=["200 per day", "50 per hour"],
    storage_uri="memory://",
)


def create_app():
    app = Flask(__name__)

    # ── SECRET_KEY hardening ─────────────────────────────────────────────
    app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY') or os.urandom(32).hex()

    app.config['UPLOAD_FOLDER'] = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'uploads'
    )

    # ── Extensions ───────────────────────────────────────────────────────
    csrf.init_app(app)
    limiter.init_app(app)

    # ── Logging (visible in docker logs) ─────────────────────────────────
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
    )

    # Ensure directories exist
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    os.makedirs(os.path.join(app.static_folder, 'charts'), exist_ok=True)

    # Register blueprint
    from app.routes import main_bp
    app.register_blueprint(main_bp)

    # ── Error handlers ───────────────────────────────────────────────────
    @app.errorhandler(404)
    def not_found(e):
        return render_template('404.html'), 404

    @app.errorhandler(500)
    def server_error(e):
        return render_template('500.html'), 500

    @app.errorhandler(429)
    def rate_limit_exceeded(e):
        return render_template('429.html'), 429

    # Initialize APScheduler for 60-minute file cleanup
    from app.utils.scheduler import init_scheduler
    init_scheduler(app)

    return app
