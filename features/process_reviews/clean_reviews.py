from features.process_text.clean import clean_text, clean_text_sentiment
import copy

def clean_review(review):
    """Cleans text of review."""
    cleaned_review = copy.copy(review)
    cleaned_review.cleaning_log, cleaned_review.reviewtext_cleaned = clean_text(cleaned_review.reviewText, tokenize=True)
    return cleaned_review


def clean_review2(review2):
    """Cleans text of review."""
    cleaned_review2 = copy.copy(review2)
    cleaned_review2.cleaning_log, cleaned_review2.reviewtext_cleaned = clean_text_sentiment(cleaned_review2.reviewText, tokenize=True)
    return cleaned_review2

def clean_review3(review3):
    """Cleans text of review."""
    cleaned_review3 = copy.copy(review3)
    cleaned_review3.cleaning_log, cleaned_review3.reviewtext_cleaned = clean_text_sentiment(cleaned_review3.reviewText, tokenize=False)
    return cleaned_review3
