"""
Play Store review scraper using google-play-scraper.
Scrapes up to 15,000 Indonesian reviews.
Sentiment labeling is handled separately via text-based scoring.
"""

import gc
import pandas as pd
from google_play_scraper import Sort, reviews


def scrape_reviews(app_id, max_reviews=15000, progress_callback=None):
    """
    Scrape reviews from the Google Play Store.

    Paginates in batches of 200 (Google Play limit per page) using
    continuation tokens. Returns raw DataFrame without sentiment labels.
    Sentiment labeling is done via text-based scoring in a separate step.

    Returns a pandas DataFrame or None if no reviews found.
    """
    all_reviews = []
    continuation_token = None
    batch_size = 200  # Google Play maximum per request

    while len(all_reviews) < max_reviews:
        remaining = max_reviews - len(all_reviews)
        count = min(batch_size, remaining)

        try:
            result, continuation_token = reviews(
                app_id,
                lang='id',
                country='id',
                sort=Sort.NEWEST,
                count=count,
                continuation_token=continuation_token,
            )
        except Exception as e:
            print(f"[Scraper] Error during scraping: {e}")
            break

        if not result:
            break

        all_reviews.extend(result)

        if progress_callback:
            progress_callback(len(all_reviews), max_reviews)

        if continuation_token is None:
            break

    if not all_reviews:
        return None

    # Keep only essential columns to save memory
    df = pd.DataFrame(all_reviews)[['content', 'score', 'userName', 'at']]
    df = df.dropna(subset=['content'])
    df = df[df['content'].str.strip() != '']
    df = df.reset_index(drop=True)

    del all_reviews
    gc.collect()
    return df
