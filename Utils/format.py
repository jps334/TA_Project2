from tabulate import tabulate


def format_reviews_table(reviews):
    """
    Formats the reviews in a tabular style.
    Args:
        reviews: A list or set of week_3.tweet objects
    Returns:
        A string formatted table of tweets
    """
    print_list = [_convert_review_to_list(review) for review in reviews]
    return tabulate(print_list, headers=["Reviewer ID", "Product ID", "Reviewer Name", "Helpfullnesss", "Rating", "Summary", "Unix Review Time", "Raw Review Time", ])

def _convert_review_to_list(review):
    """
    Converts a review to a list object
    Args:
        review: A review object

    Returns:
        A list with review information
    """

    return [review.reviewerID, review.asin, review.reviewerName, review.helpful, review.reviewText, review.overall, review.summary, review.unixReviewTime, review.reviewTime]