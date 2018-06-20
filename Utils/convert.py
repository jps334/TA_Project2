
def _clean_review_date_string(date_string):
    """
    Cleans review date string.
    Args:
        date_string: A review date string.

    Returns:
        A cleaned date string in format
    """
    date_list = date_string.split()
    year = date_list[6:10]
    month = date_list[3:5]
    day = date_list[0:2]

    return year + " " + month + " " + day