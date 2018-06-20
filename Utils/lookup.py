def get_review_by_reviewer_and_product_id(review_set, reviewer_id, asin):
    """
    Retrieves review, based on reviewer and product id id.
    Args:
        review_set: A set of reviews.
        reviewer_id: The ID of the .
        asin: The ID of the product
    Returns:
        A review with the defined reviewer and product ids.
    """
    for review in review_set:
        if review.reviewerid == reviewer_id and review.asin == asin:
            return review
