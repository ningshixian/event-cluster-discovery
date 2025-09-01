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

"""定时增量更新
每日凌晨 0 点，取末尾 3 万条的存量数据，重新进行更新？

具体步骤包括：
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
# new_data_path = "./data/new_posts.csv"  # 新帖子

# 输出结果文件
path1 = './dbscan/result/d_cluster_old.csv'
path2 = './dbscan/result/d_cluster_new.csv'
path3 = './dbscan/result/d_topic_new.csv'

# tokenizer = AutoTokenizer.from_pretrained(constants.M3E_EMBEDDING_PATH)
# model = AutoModel.from_pretrained(constants.M3E_EMBEDDING_PATH)
model = SentenceTransformer(constants.M3E_EMBEDDING_PATH, device="cuda")


# 读取存量帖子
# event_id,work_id,post
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
print("max_event_id: ", max_event_id)


"""2、存量帖子-新事件发现↓↓↓
未分配新帖子与存量-1帖子数据重新聚类，进行新事件发现
为避免存量-1帖子数量随时间不断累加，仅取末尾 3 万条存量帖子，且仅在凌晨更新？
"""
from a_post_clustering import db_scan_cluster
from b_topic_generation import cluster_topic_generate
from c_merge_cluster import cluster_merge

slice_rows = 30000
use_fast_mode =  False   
# True: 快速模式，仅针对未分配的新帖子
# False: 全量模式，针对未分配新帖子 + 存量-1帖子

# 未分配新帖子 + 存量[-30000:]中的-1帖子数据
df_cluster_stock_1 = df_cluster_stock.iloc[-slice_rows:][df_cluster_stock['event_id'] == '-1']
posts_stock_1, work_ids_stock_1 = df_cluster_stock_1["post"].tolist(), df_cluster_stock_1["work_id"].tolist()
encoded_texts_stock_1 = model.encode(posts_stock_1, batch_size=64, show_progress_bar=True) # 32266
print("存量-1噪音帖子数量：", len(posts_stock_1)) # 32266

posts = posts_stock_1
work_ids = work_ids_stock_1
encoded_texts = encoded_texts_stock_1

print("重新聚类...") # 92
db_scan_cluster(
    posts, work_ids, encoded_texts, 
    write_file_path=path2, accum_count=max_event_id
)
# input("继续...")

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
print("新事件描述生成结果保存在：", path3)

# # 聚簇合并
# cluster_merge(path2, path3, path3, accum_count=max_event_id)

"""3、寻找相似事件，映射为存量事件↓↓↓
"""
print("\n寻找相似事件，映射为存量事件...")
df3 = []    # 用于更新new事件
id2mergeid = {'-1':'-1'}    # 用于更新new聚簇结果中的event_id，old索引映射到new索引

if not df2.empty:   # 没有新事件
    df2 = pd.read_csv(path3, encoding="utf-8", keep_default_na=False)
    vec_hist_summary = model.encode(event_list, batch_size=64, show_progress_bar=True)

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
# input("继续...")

"""
4、存量&增量结果合并↓↓↓
"""
# use_fast_mode = False
print("\n由于存量-1帖子数据被用于新事件重新发现，因此需要过滤掉！")
df_first_part = df_cluster_stock.iloc[:-slice_rows]
df_last_part = df_cluster_stock.iloc[-slice_rows:][df_cluster_stock['event_id'] != '-1']
df_cluster_stock = pd.concat([df_first_part, df_last_part], axis=0)
print("存量帖子数量：", df_cluster_stock.shape)
print(df_cluster_stock.head())

# 清空 d_cluster_old 数据
df_old = pd.read_csv(path1, encoding="utf-8", keep_default_na=False)
df_old.drop(df_old.index, inplace=True)
df_old.to_csv(path1, index=False)

# 合并重新聚类的帖子
df_new = pd.read_csv(path2.replace(".csv", "_update.csv"), encoding="utf-8", keep_default_na=False)
df_new = df_new.astype(str)
df = pd.concat([df_cluster_stock, df_new], axis=0)
df.to_csv(cluster_file_path, index=False)
# df.to_csv("./dbscan/result/event_mapping_table.csv", index=False)

# 合并重新聚类的事件
df_hist = pd.read_csv(merge_file_path, encoding="utf-8", keep_default_na=False)
df_hist = df_hist.astype(str)
df_new = pd.read_csv(path3, encoding="utf-8", keep_default_na=False)
df_new = df_new.astype(str)
df = pd.concat([df_hist, df_new], axis=0)
df.to_csv(merge_file_path, index=False)
# df.to_csv("./dbscan/result/event_details_table.csv", index=False)
# input("继续...")


from e_export import incremental_update
# 增量更新（use_fast_mode=False）
increment_posts_path = './dbscan/result/d_cluster_new_update.csv'    # 新聚簇结果
increment_events_path = './dbscan/result/d_topic_new.csv'    
producer_data = incremental_update(increment_posts_path, increment_events_path)
# df.row = work_id, event_id, event

# # 发送 kafka 消息（不行，无法获取work_id对应的data_json内容）
# from e_export import publish_to_kafka
# print(f"发送清洗后数据 → kafka...({len(producer_data)})")
# for _, row in producer_data.iterrows():
#     publish_to_kafka(
#         msg_ori[row['work_id']]['data_json'], 
#         row["work_id"], 
#         row["event_id"], 
#         row["event"]
#     )
