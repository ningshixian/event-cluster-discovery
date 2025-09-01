import os
os.environ["CUDA_VISIBLE_DEVICES"] = "3"
import time
import torch
import pandas as pd
from collections import defaultdict
import sys
sys.path.append("..")
from common import constants
from transformers import AutoTokenizer, AutoModel
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn import metrics

from textrank4zh import TextRank4Keyword, TextRank4Sentence
# import jionlp as jio
from gpt4o_api import generate
import pandas as pd
from tqdm import tqdm

"""
1. 读取清洗好的帖子数据，拼接【标题+正文内容+ocr内容】，进行向量化（stella-large）
2. 使用DBSCAN聚类算法，无需指定聚簇个数
  - 参数设置为eps=0.2, min_samples=6，并使用余弦相似度作为距离度量方法。
"""

# data_file_path = './data/test_posts.csv'
# cluster_file_path = './dbscan/result/test_cluster.csv'
# topic_file_path = './dbscan/result/test_topic_summary.csv'
# merge_file_path = './dbscan/result/test_merge.csv'

data_file_path = './data/posts.csv'
cluster_file_path = './dbscan/result/cluster_result.csv'
topic_file_path = './dbscan/result/topic_summary.csv'
merge_file_path = './dbscan/result/topic_summary_merge.csv'


def get_embedding_data(file_path):
    df = pd.read_csv(
        file_path, 
        encoding="utf-8", 
        keep_default_na=False, 
        on_bad_lines='skip',
        # nrows=640,
    )
    df = df.astype(str)
    df = df.rename(columns={'work_title': '标题', 'work_content': '内容', 'ocr_content': 'ocr内容'})
    print("待聚类帖子数量：", df.shape)  # 7781 → 47984 → 191601

    posts, work_ids = [], []
    # 拼接标题+正文内容+ocr内容，注意去掉重复和空
    for i,row in df.iterrows():
        if row['内容'].startswith(row['标题']):
            row['标题'] = ""
        if row["source"] not in ["抖音 APP", "快手 APP"]:
            row['ocr内容'] = ""
        parts = [part for part in [row['标题'].replace('\n', '').strip(), row['内容'].replace('\n', '').strip(), row['ocr内容'].replace('\n', '').strip()] if part]
        key = " || ".join(parts)
        posts.append(key)
        work_ids.append(row.get('work_id', ''))

    print("待聚类帖子数量-去重：", len(posts))  # 5682 → 47123 → 191601
    return posts, work_ids


def db_scan_cluster(posts, work_ids, encoded_texts, write_file_path, accum_count=0):
    """
    对给定的帖子内容进行DBSCAN聚类分析，并将结果保存到指定路径。
    目标：压缩聚簇个数，保证簇内高聚合

    参数:
    - posts: list, 包含帖子内容的列表。
    - work_ids: list, 包含帖子唯一标识的列表。
    - encoded_texts: numpy.ndarray, 已经编码的文本向量。
    - write_file_path: str, 聚类结果保存的文件路径。
    - accum_count: int, 聚类ID的累积偏移量，默认为0。

    功能:
    - 使用StandardScaler对编码后的文本向量进行标准化。
    - 应用DBSCAN算法进行聚类，参数设置为`eps=0.2`, `min_samples=6`，并使用余弦相似度作为距离度量方法。
    - 计算并输出聚类数量及噪声点数量。
    - 将聚类结果保存到CSV文件中，包含`event_id`（聚类ID）、`work_id`（帖子ID）和`post`（帖子内容）三列。
    - 输出聚类结果文件路径及处理耗时。
    """
    scaler = StandardScaler()
    embeddings_scaled = scaler.fit_transform(encoded_texts)
    
    # while True:
    #     eps = float(input("ϵ-邻域半径："))  # 0.40 0.2
    #     min_samples = int(input("邻域内最少样本数："))  # 12 6
    #     if eps == "" or min_samples == "":
    #         break

    # DBSCAN 聚类参数
    eps = 0.2
    min_samples = 6

    # DBSCAN 聚类
    # 距离度量方法采用余弦相似度进行度量，Eps 参数设置 0.3～0.5, min_samples不能太小，否则簇合并工作量大！
    # 噪声太大==>增大eps，缩小min_sample；太少反过来
    # 希望精准度高，调低邻域半径eps；希望聚类数量多一些，调低最少样本数min_samples！！！
    start_time = time.time()
    dbscan = DBSCAN(eps=eps, min_samples=min_samples, metric='cosine').fit(embeddings_scaled)
    labels = dbscan.labels_
    labels = list(map(lambda x:x+accum_count if x!=-1 else x, labels))

    # print(dbscan.core_sample_indices_)  # 核心样本的索引
    # print(dbscan.components_)  # 通过训练找到的每个核心样本的副本
    # print(dbscan.labels_)  # 提供fit()的数据集中每个点的聚类标签。有噪声的样本被赋予标签-1。
    
    # 计算聚类数量和噪声点数量（聚类结果中-1表示没有聚类为离散点）
    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise_ = list(labels).count(-1)
    print('得到的聚类个数为: %d' % n_clusters_)  # 1644
    print('Estimated number of noise points: %d' % n_noise_)    # 108026
    print("耗时：", time.time() - start_time)   # 2681s ≈ 44min

    # # 模型评估
    # print("轮廓系数: %0.3f" % metrics.silhouette_score(embeddings_scaled, labels))  # 0.326
    
    result = pd.DataFrame(columns=['event_id', 'work_id', 'post'])
    result['event_id'] = labels
    result['work_id'] = work_ids
    result['post'] = posts

    # 统计聚类数量
    cluster_count = result['event_id'].value_counts()
    print(cluster_count)

    # 保持相同cluster ID的行在一起
    result = result.groupby('event_id').apply(lambda x: x).reset_index(drop=True)
    # 聚类结果输出csv
    result.to_csv(write_file_path, index=False)
    print("聚类结果保存在：", write_file_path)      #  ./dbscan/result/cluster_result.csv
    print("耗时：", time.time() - start_time)     # 2693s ≈ 45min


if __name__ == "__main__":
    
    # tokenizer = AutoTokenizer.from_pretrained(constants.M3E_EMBEDDING_PATH)
    # model = AutoModel.from_pretrained(constants.M3E_EMBEDDING_PATH)
    model = SentenceTransformer(constants.M3E_EMBEDDING_PATH, device="cuda")

    """1、帖子向量化，使用DBSCAN聚类
    目标：压缩聚簇个数，保证簇内高聚合
    """
    posts, work_ids = get_embedding_data(data_file_path)
    encode_posts = model.encode(posts, batch_size=64, show_progress_bar=True)  # normalize_embeddings ✖
    db_scan_cluster(posts, work_ids, encode_posts, cluster_file_path)
