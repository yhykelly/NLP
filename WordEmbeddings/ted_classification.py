from gensim.models import KeyedVectors
import spacy
import numpy as np
from numpy import dot
from numpy.linalg import norm # used to calculate normalised vector from v to |v|
import pandas as pd

#
# -- Important Note -- 
# Several functions have been deprecated in the
# latest scipy package (1.13).
# Since the gensim package depends on scipy, you will need to
# install an earlier version of scipy (1.11.4)
#

# ToDo: load the spacy english small model,
#  disable parser and ner pipes
print('Loading spaCy...')
nlp = spacy.load("en_core_web_sm")

# ToDo: load word2vec embeddings,
#  which are located in the same directory as this script.
#  Limit the vocabulary to 100,000 words
#  - should be enough for this application
#  - loading all takes a long time
print('Loading word2vec embeddings...')

emb = KeyedVectors.load_word2vec_format(
            'GoogleNews-vectors-negative300.bin.gz',
            binary=True,
            limit=200000)


def load_data(filename):
    """
    Load the Ted Talk data from filename and extract the
    "description" and "url" columns. Return a dictionary of dictionaries,
    where the keys of the outer dictionary are unique integer values which
    will be used as IDs.
    Each inner dictionary represents a row in the input file and contains
    the keys 'description', and 'url'.

    :param filename: input filename in csv format
    :return: dict of dicts, where the inner dicts represent rows in the input
    file, and the keys in the outer dict serve as IDs.
    """

    df = pd.read_csv(filename)
    df2 = df[['description', 'url']]
    dict2 = df2.T.to_dict()
    return dict2


def preprocess_text(text):
    """
    Preprocess one text. Helper function for preprocess_texts().

    Preprocessing includes lowercasing and removal of stopwords,
    punctuation, whitespace, and urls.

    The returned list of tokens could be an empty list.

    :param text: one text string
    :return: list of preprocessed token strings. May be an empty list if all tokens are eliminated.
    """

    tokens = []
    doc = nlp(text)
    for token in doc:
        reject = token.like_url or token.is_stop or token.is_punct or token.is_space
        if (not reject):
            tokens.append(token.text)

    tokens = [token.lower() for token in tokens]
    return tokens


def preprocess_texts(data_dict):
    """
    Preprocess the description in each inner dict of data_dict by
    lowercasing and removing stopwords, punctuation, whitespace, and urls.
    The list of token strings for an individual text is not a set,
    and therefore may contain duplicates. Add a new key 'pp_text'
    to each inner dict, where the value is a list[str] representing
    the preprocessed tokens the text.

    :param data_dict: a nested dictionary with a structure as returned by load_data()
    :return: the input data_dict, with key 'pp_text' of preprocessed token strings for
    each description
    """

    for index, data in data_dict.items():
        text = data['description']
        result = preprocess_text(text)
        data['pp_text'] = result
    
    return data_dict


def get_vector(tokens):
    """
    Calculate a single vector for the preprocessed word strings in tokens.
    The vector is calculated as the mean of the word2vec vectors for the
    words in tokens. Words that are not contained in the word2vec pretrained
    embeddings are ignored. If none of the tokens are contained in word2vec,
    return None.

    :param tokens: list of strings containing the preprocessed tokens
    :return: mean of the word2vec vectors of the words in tokens, or None
    if none of the tokens are in word2vec.
    """

    if tokens == []:
        return None
    else:
        aggregate = []
        for token in tokens:
            if token in emb:
                embedding = emb[token]
                aggregate.append(embedding)
            else:
                continue

    if aggregate != []:
        np_aggregate = np.array(aggregate)
        mean = np.mean(np_aggregate, axis=0)
        return mean
    else:
        return None


def get_vectors(data_dict):
    """
    Calculate the vector of the preprocessed text 'pp_text' in each
    inner dict of data_dict. Add a new key 'vector'
    to each inner dict, where the value is the mean of individual word vectors
    as returned by get_vector().

    If 'pp_text' is an empty list, or none of the words in 'pp_text' are
    in word2vec, the value of 'vector' is None.

    :param data_dict: a nested dictionary where inner dicts have key 'pp_text'
    :return: the input data_dict, with key 'vector' for each inner dict
    """

    for _, data in data_dict.items():
        text_list = data['pp_text']
        result = get_vector(text_list)
        data['vector'] = result
    
    return data_dict


def cosine_similarity(v1, v2):
    """
    Calculate the cosine similarity of v1 and v2.

    :param v1: vector 1
    :param v2: vector 2
    :return: cosine similarity
    """

    dot_product = np.dot(v1, v2)
    v1_norm = norm(v1)
    v2_norm = norm(v2)
    total_norm = v1_norm * v2_norm
    cos = dot_product / total_norm
    return cos


def k_most_similar(query, data_dict, k=5):
    """
    Find the k most similar entries in data_dict to the query.

    The query is first preprocessed, then a mean word vector is calculated for it.

    Return a list of tuples of length k where each tuple contains
    the id of the data_dict entry and the cosine similarity score between the
    data_dict entry and the user query.

    In some cases, the similarity cannot be calculated. For example:
    - If none of the preprocessed token strings are in word2vec for an entry in data_dict.
    If you built data_dict according to the instructions, the value of 'vector'
    is None in these cases, and those entries should simply not be considered.
    - If a vector for the query can't be calculated, return an empty list.

    :param query: a query string as typed by the user
    :param data_dict: a nested dictionary where inner dicts have key 'vector'
    :param k: number of top results to return
    :return: a list of tuples of length k, each containing an id and a similarity score,
    or an empty list if the query can't be processed
    """

    query_tokens = preprocess_text(query) # get the tokens list of query
    query_vector = get_vector(query_tokens) # get the vector of the query
    results = []
    if np.array(query_vector).size == 1: # if query_tokens = [] -> NoneType size = 1
        return results
    else: # normal vector size == 300
        for index, data in data_dict.items():
            data_vector = data['vector']
            if not data_vector.all() == None:
                result = cosine_similarity(data_vector, query_vector)
                pair = (index, result)
                results.append(pair)
            else:
                continue
    
    results_sorted = sorted(results, key=lambda x: x[1], reverse=True)
    return results_sorted[:k]


def recommender_app(data_dict):
    """
    Implement your recommendation system here.

    - Repeatedly prompt the user to type a query
        - Print the description and the url of the top 5 most similar,
        or "No Results Found" if appropriate
        - Return when the query is "quit" (without quotes)

    :param data_dict: nested dictionaries containing
    description,url,tokens,and vectors for each description
    in the input data
    """

    while True:
        query = input("type a query for TED recommendation: ")
        if query == "quit":
            break
        else:
            result = k_most_similar(query, data_dict, k=5)

        if result == []:
            print("\n No Results Found \n")
        else:
            print("\n Query results: \n ------------- \n ")
            for index, _ in result:
                print(data_dict[index]['description'])
                print(data_dict[index]['url'])
                print()
            print()


def main():
    """
    Bring it all together here.
    """
    data_dict = load_data("ted_main.csv")
    data_dict = preprocess_texts(data_dict)
    data_dict = get_vectors(data_dict)
    recommender_app(data_dict)


if __name__ == '__main__':
    main()
