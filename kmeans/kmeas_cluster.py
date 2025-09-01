
import re
import pandas as pd
import jieba
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer

from common import constants


def load_stopwords(file_path):
    stopword = []
    with open(file_path, 'r', encoding=None) as file:
        for word in file.readlines():
            stopword.append(word.strip())
    return list(set(stopword))


stopwords = load_stopwords(constants.STOPWORDS_PATH)


def get_kmeans_data(file_path):
    df = pd.DataFrame(columns=['query'])
    topic_set = set()
    with open(file_path, 'r') as f:
        for line in f.readlines():
            topic_set.add(line.strip())
    df['query'] = list(topic_set)
    return df


def word_filter(query):
    line = re.sub(r'[a-zA-Z0-9]*', '', query)
    line.replace(' ', '')
    wordlist = jieba.lcut(line, cut_all=False)
    return ' '.join([word for word in wordlist if word not in stopwords and len(word) > 1])


def query_preprocess(df):
    return list(df['query'].apply(word_filter))


def transform(dataset, n_features=256):
    do_vector = TfidfVectorizer(max_df=0.7, max_features=n_features, min_df=0.01,
                                use_idf=True, smooth_idf=True, lowercase=False, analyzer='word')
    fit = do_vector.fit_transform(dataset)
    return fit, do_vector


def k_means(df):

    dataset = query_preprocess(df)

    fit, do_vector = transform(dataset, n_features=256)
    true_k = 30
    km = KMeans(n_clusters=true_k, init='k-means++', max_iter=300, n_init=50, verbose=False)
    km.fit(fit)

    cluster_assignment = km.labels_

    df['kmeans-label'] = cluster_assignment

    return df


if __name__ == '__main__':
    df_data = get_kmeans_data('../data/topics.txt')
    df_labeled_data = k_means(df_data)

    for index, row in df_labeled_data[df_labeled_data['kmeans-label'] == 28].iterrows():
        print(index)
        print(row)
        if index >= 5000:
            break
