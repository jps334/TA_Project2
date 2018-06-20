
from sklearn.feature_extraction.text import TfidfVectorizer



def tfidf_bow(dataset):
    """
    Args:
        dataset: a collection of documents stored in a vector
    Returns:
        A list of words, corresponding to the indexed vocabulary of the dataset
    """
    vectorizer = TfidfVectorizer()
    x = vectorizer.fit_transform(dataset)

    return x, vectorizer




