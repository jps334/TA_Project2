

class Review(object):
    """Class represents review from Amazon.

    Attributes:
        reviewerID - ID of the reviewer, e.g. A2SUAM1J3GNN3B
        asin - ID of the product, e.g. 0000013714
        reviewerName - name of the reviewer
        helpful - helpfulness rating of the review, e.g. 2/3
        reviewText - text of the review
        overall - rating of the product
        summary - summary of the review
        unixReviewTime - time of the review (unix time)
        reviewTime - time of the review (raw)
    """

    def __init__(self, reviewerID, asin, reviewerName,  helpful, reviewText, overall,summary, reviewTime, Category):
        """Initializes review object with defined content."""
        self.reviewerID = reviewerID
        self.asin = asin
        self.reviewerName = reviewerName
        self.helpful = helpful
        self.reviewText = reviewText
        self.overall = overall
        self.summary = summary
        self.reviewTime = reviewTime
        self.reviewtext_cleaned = None
        self.cleaning_log = dict()
        self.category = Category

    def __iter__(self):
        return self


    def __str__(self):
        """Creates user-friendly string representation of reviews."""

        return "ReviewerID: {}\nasin: {}\nreviewerName: {}\nhelpfulness: {}\noverall: {}\nsummary:{}\nreviewTime: {}\nreviewText:\n{}".\
            format(self.reviewerID, self.asin, self.reviewerName, self.helpful,
                   self.overall, self.summary,  self.reviewTime, self.reviewText, self.category)


def create_review_from_dict(review_dict):
    """
    Creates a review object from dictionary.

    Extracts reviewerID, asin, text, is_retweet,
    retweet_count and favorite_count from dictionary.

    Args:
        review_df: A dataframe, containing review information.

    Returns:
        A review object.
    """
    # Extract parameters from dataframe
    reviewerid = review_dict.get('reviewerID')
    reviewerName = review_dict.get('reviewerName')
    helpful = review_dict.get('helpful')
    reviewText = review_dict.get('reviewText')
    overall = review_dict.get('overall')
    summary = review_dict.get('summary')
    asin = review_dict.get('asin')
    reviewTime = review_dict.get('reviewTime')
    category =  review_dict.get('Category')
    # Create week_3.tweet object
    review = Review(reviewerid, asin, reviewerName, helpful, reviewText, overall, summary, reviewTime, category)

    return review
