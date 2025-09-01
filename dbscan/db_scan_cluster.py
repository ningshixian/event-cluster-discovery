import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
import torch
import pandas as pd
from collections import defaultdict
import sys
sys.path.append("..")
from common import constants
from transformers import AutoTokenizer, AutoModel
from sentence_transformers import SentenceTransformer
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn import metrics
import matplotlib.pyplot as plt
import seaborn as sns  # 可视化库
sns.set(font='SimHei', style='ticks')
from tqdm import tqdm
import numpy as np  
import pickle

"""1、帖子向量化，使用DBSCAN聚类
目标：压缩聚簇个数，保证簇内高聚合
"""

# tokenizer = AutoTokenizer.from_pretrained(constants.M3E_EMBEDDING_PATH)
# model = AutoModel.from_pretrained(constants.M3E_EMBEDDING_PATH)
model = SentenceTransformer(constants.M3E_EMBEDDING_PATH, device="cuda")

def get_embedding_data(file_path):
    topic_set = set()
    df = pd.read_csv(
        file_path, 
        encoding="utf-8", 
        keep_default_na=False, 
        on_bad_lines='skip',
    )
    for i,row in df.iterrows():
        topic_set.add(row['标题'] + " || " + row['内容'])

    print("待聚类帖子数量topic_set：", len(topic_set))  # 3779
    return list(topic_set)


# def embedding_topic(topic_list):
#     encoded_texts = []
#     for text in tqdm(topic_list, total=len(topic_list)):
#         # inputs = tokenizer(text, return_tensors="pt")
#         inputs = tokenizer(text, padding=True, truncation=True, max_length=512, return_tensors="pt")
#         outputs = model(**inputs)

#         last_hidden_states = outputs.last_hidden_state
#         text_vector = torch.mean(last_hidden_states, dim=1).squeeze().detach().cpu().numpy()
#         encoded_texts.append(text_vector)

#     return encoded_texts


def embedding_topic(topic_list):
    vectors = model.encode(topic_list, batch_size=64, show_progress_bar=True)  # normalize_embeddings ✖
    print(vectors.shape)
    # # vectors 可以提前计算并保存
    # with open('../data/vectors.pkl', 'wb') as f:
    #     pickle.dump(vectors, f)
    # # 加载pkl文件中的数据（效果变差？）
    # with open('../data/vectors.pkl', 'rb') as f:
    #     vectors = pickle.load(f)
    return vectors  


def db_scan_cluster(topic_list, encoded_texts, write_file_path):
    scaler = StandardScaler()
    embeddings_scaled = scaler.fit_transform(encoded_texts)

    # DBSCAN 聚类
    # 注意针对不同任务进行 eps、min_samples 的调参，不同参数配置差异较大
    # 距离度量方法采用余弦相似度进行度量，Eps 参数设置 0.3～0.5, min_samples不能太小，否则簇合并工作量大！
    # 噪声太大==>增大eps，缩小min_sample；太少反过来
    dbscan = DBSCAN(eps=0.40, min_samples=12, metric='cosine')    # 轮廓系数: 0.180
    labels = dbscan.fit_predict(embeddings_scaled)

    # print(dbscan.labels_)  # db.labels_为所有样本的聚类索引，没有聚类索引为-1
    # print(dbscan.core_sample_indices_) # 所有核心样本的索引
    # print(dbscan.components_)  # 所有核心样本 n_neighbors >= self.min_samples
    labels = dbscan.labels_
    
    # 获取聚类个数（聚类结果中-1表示没有聚类为离散点）
    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
    print('估计的聚类个数为: %d' % n_clusters_)

    # 模型评估
    # print("同质性: %0.3f" % metrics.homogeneity_score(labels_true, labels))  # 每个群集只包含单个类的成员。
    # print("完整性: %0.3f" % metrics.completeness_score(labels_true, labels))  # 给定类的所有成员都分配给同一个群集。
    # print("V-measure: %0.3f" % metrics.v_measure_score(labels_true, labels))  # 同质性和完整性的调和平均
    # print("调整兰德指数: %0.3f" % metrics.adjusted_rand_score(labels_true, labels))
    # print("调整互信息: %0.3f" % metrics.adjusted_mutual_info_score(labels_true, labels))
    print("轮廓系数: %0.3f" % metrics.silhouette_score(embeddings_scaled, labels))  # 0.326
    
    # # 画布设置
    # fig = plt.figure(figsize=(12, 5))
    # fig.subplots_adjust(left=0.02, right=0.98, bottom=0.05, top=0.9)
    # ax = fig.add_subplot(1, 2, 2)
    # unique_labels = set(labels)
    # colors = [plt.cm.Spectral(each) for each in np.linspace(0, 1, len(unique_labels))]
    # core_samples_mask = np.zeros_like(dbscan.labels_, dtype=bool)  # 设置一个样本个数长度的全false向量
    # core_samples_mask[dbscan.core_sample_indices_] = True  # 将核心样本部分设置为true
    # for k, col in zip(unique_labels, colors):
    #     if k == -1:  # 聚类结果为-1的样本为离散点
    #         # 使用黑色绘制离散点
    #         col = [0, 0, 0, 1]
    
    #     class_member_mask = (labels == k)  # 将所有属于该聚类的样本位置置为true
    #     xy = X[class_member_mask & core_samples_mask]  # 将所有属于该类的核心样本取出，使用大图标绘制
    #     ax.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col), markeredgecolor='k', markersize=14)
    #     xy = X[class_member_mask & ~core_samples_mask]  # 将所有属于该类的非核心样本取出，使用小图标绘制
    #     ax.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col), markeredgecolor='k', markersize=6)
    
    # plt.title('Estimated number of clusters: %d' % n_clusters_)
    # sns.despine()
    # plt.show()

    result = pd.DataFrame(columns=['query', 'cluster_label'])
    result['query'] = topic_list
    result['cluster_label'] = labels

    cluster_t_dict = defaultdict(list)
    for t, cluster_id in zip(topic_list, labels):
        cluster_t_dict[cluster_id].append(t)

    # 聚类结果输出
    with open(write_file_path, 'w+') as f:
        for cluster_id, topic in cluster_t_dict.items():
            topic = "\n".join(topic)
            f.write(f'{cluster_id}\n{topic}\n\n')


if __name__ == '__main__':
    topics = get_embedding_data('../data/posts_test.csv')
    encode_topics = embedding_topic(topics)
    write_file_path = './result/cluster_result_09_02.txt'
    db_scan_cluster(topics, encode_topics, write_file_path)
