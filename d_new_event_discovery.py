import warnings
warnings.filterwarnings('ignore')
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "3"
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd

import sys
sys.path.append("..")
from common import constants

"""
4、新帖聚类&新事件发现
此模块实现了对新帖子的存量事件自动分配和新事件发现。

具体步骤包括：
1. 新帖子分配存量事件
    - 计算新帖子与存量事件描述之间的相似度
    - 根据设定的阈值将新帖子分配给最相似的存量事件或标记为未知事件
2. 新事件发现：
    - 对未分配的新帖子进行聚类分析，发现新的事件（可以和存量的-1噪音数据合并重新聚类，不过太费时间，可以凌晨跑一次）。 
    - 生成新事件的标题和描述。
3. 新事件与存量事件匹配：
    - 检查新发现的事件是否与存量事件高度相似。
    - 如果相似度超过一定阈值，则将新事件映射为对应的存量事件。
4. 结果整合：
    - 合并所有聚类结果，包括存量帖子、已分配的新帖子以及新发现的事件。
    - 更新事件汇总表，添加新发现的事件信息。
"""

cluster_file_path = './dbscan/result/cluster_result_merge.csv'  # 存量帖子聚簇结果
merge_file_path = './dbscan/result/topic_summary_merge.csv'     # 存量事件
new_data_path = "./data/new_posts.csv"  # 新帖子

# 输出结果文件
path1 = './dbscan/result/d_cluster_old.csv'
path2 = './dbscan/result/d_cluster_new.csv'
path3 = './dbscan/result/d_topic_new.csv'

# tokenizer = AutoTokenizer.from_pretrained(constants.M3E_EMBEDDING_PATH)
# model = AutoModel.from_pretrained(constants.M3E_EMBEDDING_PATH)
model = SentenceTransformer(constants.M3E_EMBEDDING_PATH, device="cuda")


# 读取存量帖子
df_cluster_stock = pd.read_csv(cluster_file_path, encoding="utf-8", keep_default_na=False)
df_cluster_stock = df_cluster_stock.astype(str) # 所有元素数据类型统一为str
# # 找出work_id 等于特定值的行
# df_cluster_stock = df_cluster_stock[df_cluster_stock['work_id'] == 'b048d17603f77017b2f42f22cc95db04']

# 读取存量事件
df_hist = pd.read_csv(
    merge_file_path, 
    encoding="utf-8", 
    keep_default_na=False  # 空值都是默认为 NAN，设置 keep_default_na=False 让读取出来的空值是空字符串
)
df_hist = df_hist.astype(str)
event_list = [f"事件标题：{title}\n概要描述：{description}" for title, description in zip(df_hist['event_title'], df_hist['event_description'])]
event2id = {event:event_id for event,event_id in zip(event_list, df_hist['event_id'])}
max_event_id = df_hist['event_id'].astype(int).max() + 1    # 存量事件的数量


# 读取新帖子
from a_post_clustering import get_embedding_data
posts_new, work_ids_new = get_embedding_data(new_data_path)
# 找出work_id重复的索引
dup_idx_set = set([i for i, wid in enumerate(work_ids_new) if wid in df_cluster_stock["work_id"].tolist()])
posts_new = [post for i, post in enumerate(posts_new) if i not in dup_idx_set]
work_ids_new = [work_id for i, work_id in enumerate(work_ids_new) if i not in dup_idx_set]
print(f"删除work_id重复的帖子，剩余{len(posts_new)} 新帖子")
# print('b048d17603f77017b2f42f22cc95db04' in df_cluster_stock["work_id"].tolist())  # False
# print('b048d17603f77017b2f42f22cc95db04' in work_ids_new)                 # True
# exit()

"""1、新帖子-分配存量事件↓↓↓
"""
print("\n=======================1、新帖子分配存量事件↓↓↓=======================")

vec_hist_summary = model.encode(event_list, batch_size=64, show_progress_bar=True)  # 388
vec_new_posts = model.encode(posts_new, batch_size=64, show_progress_bar=True)  # 1w+
sim_matrix = cosine_similarity(vec_new_posts, vec_hist_summary)  # 将新帖子与历史话题描述计算相似度
print("similar_matrix: ", sim_matrix.shape) # (10160, 388)

# 针对每个新帖子，找出sim矩阵中最相似的话题
# 当相似度大于预设的阈值时，则视为重合，则进行合并
new_event_ids, new_work_ids, new_works = [], [], []
posts_new_1, work_ids_new_1, encoded_texts_new_1 = [], [], []   # 未知事件的新帖子，待二次聚类
thredshold = 0.8
for i in range(len(posts_new)):
    max_sim_idx = sim_matrix[i].argmax()    # 分值最大的topic索引
    if sim_matrix[i][max_sim_idx] > thredshold:
        new_event_ids.append(event2id[event_list[max_sim_idx]])
        new_work_ids.append(work_ids_new[i])
        new_works.append(posts_new[i])
    else:
        # 无对应存量话题，则与噪音数据（-1）重新聚类，进行新事件发现？
        posts_new_1.append(posts_new[i])
        work_ids_new_1.append(work_ids_new[i])
        encoded_texts_new_1.append(vec_new_posts[i])

result = pd.DataFrame(columns=['event_id', 'work_id', 'post'])
result['event_id'] = new_event_ids
result['work_id'] = new_work_ids
result['post'] = new_works

# 保持相同event_id的行在一起
result = result.groupby('event_id').apply(lambda x: x).reset_index(drop=True)
result.to_csv(path1, index=False)   
print(f"总共 {len(result)} 篇新帖子被分配到存量事件中")    # 1077
print(f"剩余 {len(posts_new_1)} 篇未分配")              # 9083
print("新帖子-分配存量事件结果已保存：", path1)


"""2、新帖子-新事件发现↓↓↓

两种方式：
1. 未分配新帖子与存量-1帖子数据重新聚类，进行新事件发现？
    - 随着时间累积，存量-1帖子数量会不断增加，耗时会逐步增加
2. 为了加快新事件发现速度，可以不考虑存量-1帖子数据，仅针对未分配的新帖子进行聚类？
"""
print("\n=======================2、新帖子-新事件发现↓↓↓=======================")

from a_post_clustering import db_scan_cluster
from b_topic_generation import cluster_topic_generate
from c_merge_cluster import cluster_merge

use_fast_mode =  True

if use_fast_mode:
    # 仅针对未分配的新帖子
    posts = posts_new_1
    work_ids = work_ids_new_1
    encoded_texts = encoded_texts_new_1
else:
    # 未分配新帖子 + 存量-1帖子数据
    df_cluster_stock_1 = df_cluster_stock[df_cluster_stock['event_id'] == -1]
    posts_stock_1, work_ids_stock_1 = df_cluster_stock_1["post"], df_cluster_stock_1["work_id"]
    encoded_texts_stock_1 = model.encode(posts_stock_1, batch_size=64, show_progress_bar=True) # 32266
    print("存量-1噪音帖子数量：", len(posts_stock_1)) # 32266
    
    posts = [*posts_new_1, *posts_stock_1]
    work_ids = [*work_ids_new_1, *work_ids_stock_1]
    encoded_texts = [*encoded_texts_new_1, *encoded_texts_stock_1]
    
print("重新聚类...") # 92
db_scan_cluster(
    posts, work_ids, encoded_texts, 
    write_file_path=path2, accum_count=max_event_id
)

print("\n重新聚类完成，开始新事件标题/描述生成...")
# 新事件标题/描述生成（请求 gpt4o 耗时）
# df2 = pd.read_csv(path3, encoding="utf-8", keep_default_na=False)
# df2 = df2.astype(str)
id2summary = cluster_topic_generate(path2)
tmp = []
for k, v in id2summary.items():
    tmp.append([k, v["event_title"], v["event_description"]])
df2 = pd.DataFrame(tmp, columns=['event_id', 'event_title', 'event_description'])
df2 = df2.astype(str)
df2.to_csv(path3, index=False)
print("新事件 topic生成结果保存在：", path3)

# # 聚簇合并
# cluster_merge(path2, path3, path3, accum_count=max_event_id)

"""3、寻找相似事件，映射为存量事件↓↓↓
"""
print("\n=======================3、相似事件匹配，新事件映射为存量事件↓↓↓=======================")

df3 = []    # 用于更新new事件
id2mergeid = {'-1':'-1'}    # 用于更新new聚簇结果中的event_id，old索引映射到new索引

if not df2.empty:   # 如果新事件非空，执行d-3
    # df2 = pd.read_csv(path3, encoding="utf-8", keep_default_na=False)
    # vec_hist_summary = model.encode(event_list, batch_size=64, show_progress_bar=True)
    new_event_list = [f"事件标题：{title}\n概要描述：{description}" for title, description in zip(df2['event_title'], df2['event_description'])]
    vectors = model.encode(new_event_list, batch_size=64, show_progress_bar=True)  # normalize_embeddings ✖
    sim_score_matrix = cosine_similarity(vectors, vec_hist_summary)  # 将新事件与历史话题描述计算相似度
    print(sim_score_matrix.shape) # (new topics, 388)

    thredshold = 0.85
    for i in range(len(new_event_list)):
        max_sim_idx = sim_score_matrix[i].argmax()    # 分值最大的topic索引
        new_event_id = str(df2["event_id"][i])  # 新事件id
        if sim_score_matrix[i][max_sim_idx] > thredshold:
            id2mergeid[new_event_id] = df_hist['event_id'][max_sim_idx]  # 新事件id → 存量事件id
        else:
            id2mergeid[new_event_id] = str(max_event_id + len(df3))  # 新事件id → 自增
            # 新事件数据更新
            df2['event_id'].iloc[i] = str(max_event_id + len(df3))
            df3.append(df2.iloc[i].tolist())

print("更新new聚簇结果中的 event_id...")
_df = pd.read_csv(path2, encoding="utf-8", keep_default_na=False)
_df = _df.astype(str)
_df["event_id"] = _df["event_id"].map(id2mergeid)
_df.to_csv(path2.replace(".csv", "_update.csv"), index=False, encoding="utf-8")
print("更新new聚簇结果保存在：", path2.replace(".csv", "_update.csv"))

print("更新new事件结果...")
df3 = pd.DataFrame(df3, columns=['event_id', 'event_title', 'event_description'])
df3.to_csv(path3, index=False, encoding="utf-8")
print("更新事件结果保存在：", path3)


# =============================================================================== #
print("\n=======================4、存量&增量结果合并↓↓↓=======================")


# 合并所有【存量/已分配/new新事件】帖子聚簇结果
if not use_fast_mode:
    print("由于存量-1帖子数据已被用于 new新事件聚簇，因此需要过滤掉存量-1帖子数据，已过滤完毕！")
    df_cluster_stock = df_cluster_stock[df_cluster_stock['event_id'] != -1]
df_old = pd.read_csv(path1, encoding="utf-8", keep_default_na=False)
df_old = df_old.astype(str)
df_new = pd.read_csv(path2.replace(".csv", "_update.csv"), encoding="utf-8", keep_default_na=False)
df_new = df_new.astype(str)
df = pd.concat([df_cluster_stock, df_old, df_new], axis=0)
df.to_csv(cluster_file_path, index=False)
# df.to_csv("./dbscan/result/event_mapping_table.csv", index=False)

# 合并存量事件和新事件
df_hist = pd.read_csv(merge_file_path, encoding="utf-8", keep_default_na=False)
df_hist = df_hist.astype(str)
df_new = pd.read_csv(path3, encoding="utf-8", keep_default_na=False)
df_new = df_new.astype(str)
df = pd.concat([df_hist, df_new], axis=0)
df.to_csv(merge_file_path, index=False)
# df.to_csv("./dbscan/result/event_details_table.csv", index=False)

