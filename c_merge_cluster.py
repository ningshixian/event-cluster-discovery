import os
os.environ["CUDA_VISIBLE_DEVICES"] = "3"
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

from b_topic_generation import get_prompt_and_parser

"""
3、相似聚簇合并
基于话题摘要之间的语义相似矩阵，合并相似的话题聚簇

具体步骤包括：
- 首先读取生成的聚簇&话题摘要，并计算话题两两之间的相似度。
- 接着根据设定的阈值，从相似度矩阵中识别哪些话题应该被合并。
- 构建并查集，处理相似话题的合并逻辑。
- 对于需要合并的话题组，使用GPT重新生成新的摘要。
- 最后，更新合并后的新聚簇信息，更新原始聚类结果中的事件ID，并保存
"""


# data_file_path = './data/test_posts.csv'
# cluster_file_path = './dbscan/result/test_cluster.csv'
# topic_file_path = './dbscan/result/test_topic_summary.csv'
# merge_file_path = './dbscan/result/test_merge.csv'

data_file_path = './data/posts.csv'
cluster_file_path = './dbscan/result/cluster_result.csv'
topic_file_path = './dbscan/result/topic_summary.csv'
merge_file_path = './dbscan/result/topic_summary_merge.csv'

# tokenizer = AutoTokenizer.from_pretrained(constants.M3E_EMBEDDING_PATH)
# model = AutoModel.from_pretrained(constants.M3E_EMBEDDING_PATH)
model = SentenceTransformer(constants.M3E_EMBEDDING_PATH, device="cuda")


class UnionFindSet:
    def __init__(self, n):
        self.parent_list = [i for i in range(n)]	# 保存节点的父节点
        self.ranks = [1 for i in range(n)]	# 保存父节点的大小
        
    def find(self, node):
        """非递归版本find(x) + 路径压缩"""
        r = node # 找到node的根节点
        while r != self.parent_list[r]: # 如果假设不成立（r不是根节点），就继续循环
            # 优化：路径压缩
            self.parent_list[r] = self.parent_list[self.parent_list[r]]
            r = self.parent_list[r] # 假设根节点是当前节点的父节点，即往树的上面走一层
        return r

    def union(self, nodea, nodeb):
        """根据Rank来合并(Union by Rank)"""
        rootX,rootY = self.find(nodea), self.find(nodeb)
        if rootX==rootY:
            return
        #取rank值小的那个挂到大的那个节点下面
        #被挂的那个根节点的rank值需要+
        if(self.ranks[rootX]>self.ranks[rootY]): 
            self.parent_list[rootY] = rootX 
            self.ranks[rootX] += self.ranks[rootY]
        else: 
            self.parent_list[rootX] = rootY
            self.ranks[rootY] += self.ranks[rootX]


def cluster_merge(cluster_file_path, topic_file_path, merge_file_path, accum_count=0):
    # 读取聚簇&话题结果
    """
    合并相似的话题聚簇

    ### 参数说明
    - `cluster_file_path` (str): 聚类结果文件路径。
    - `topic_file_path` (str): 话题数据文件路径。
    - `merge_file_path` (str): 合并后的话题结果输出路径。
    - `accum_count` (int, optional): 用于合并ID的累积计数，默认为0。

    ### 主要步骤
    - 首先读取生成的聚簇话题摘要，并计算话题两两之间的相似度。
    - 接着根据设定的阈值，从相似度矩阵中识别哪些话题应该被合并。
    - 构建并查集，处理相似话题的合并逻辑。
    - 对于需要合并的话题组，使用GPT重新生成新的摘要。
    - 最后，更新合并后的新聚簇信息，更新原始聚类结果中的事件ID，并保存
    """
    df = pd.read_csv(
        topic_file_path, 
        encoding="utf-8", 
        keep_default_na=False  # 空值都是默认为 NAN，设置 keep_default_na=False 让读取出来的空值是空字符串
    )
    df = df.astype(str)
    event_list = [f"事件标题：{title}\n概要描述：{description}" for title, description in zip(df['event_title'], df['event_description'])]
    vectors = model.encode(event_list, batch_size=64, show_progress_bar=True)  # normalize_embeddings ✖
    # the output will be the pairwise similarities between all samples in X.
    sim_matrix = cosine_similarity(vectors)

    df_merge = []
    thredshold = 0.85   # 阈值0.75
    for i in range(len(sim_matrix)):
        for j in range(i+1, len(sim_matrix)):
            if sim_matrix[i][j] > thredshold:
                # print(i, j, sim_matrix[i][j])
                df_merge.append([i, j, sim_matrix[i][j]])
    
    # 通过并查集合并同类项
    uf = UnionFindSet(len(sim_matrix))
    for k in range(len(df_merge)):
        uf.union(df_merge[k][0], df_merge[k][1])
    
    # 输出 uf 中所有相同父节点的为一组
    # 创建一个字典，键为父节点（也即），值为需要合并的索引列表
    index_groups = {}
    for index, p_value in enumerate(uf.parent_list):  # index 位置对应父节点的值为 value
        if p_value not in index_groups:
            index_groups[p_value] = []
        index_groups[p_value].append(index)
    print("合并后的聚簇结果如下：", index_groups.values())
    print("其中，包含需合并的数量：", len([x for x in index_groups.values() if len(x)>1]))
    # exit()

    # ============================================================================ #

    # 输出合并后的聚簇
    id2mergeid = {"-1":"-1"}    # 原始event_id到合并后新event_id的映射
    mergeid2event = {}      # 新event_id到话题摘要的映射
    prompt, output_parser = get_prompt_and_parser("prompt_c.txt")  # 获取 Prompt 和 解析器
    for mergeid, group in tqdm(enumerate(index_groups.values()), total=len(index_groups)):
        mergeid = mergeid + accum_count # 新event_id，每次迭代累加1
        # group 存储了需合并的索引列表
        if len(group)>1:
            key_sentences = "\n".join([f"{i+1}、{ab}"for i,ab in enumerate([event_list[i] for i in group])])
            # 借助 GPT4o 模型生成话题摘要
            user_prompt = prompt.format(key_sentences=key_sentences)
            # print(user_prompt)
            llm_output = generate(user_prompt)
            # 使用解析器进行解析生成的内容
            llm_output = output_parser.parse(llm_output)
            topic_summary = f"事件标题：{llm_output.Event_Title}\n概要描述：{llm_output.Summary_Description}"
            # cluster_merge[str(group)] = topic_summary
            mergeid2event[mergeid] = {
                "event_title": llm_output.Event_Title,
                "event_description": llm_output.Summary_Description
            }
            for g in group:
                # cluster_merge2[g] = topic_summary
                id2mergeid[str(g+accum_count)] = mergeid
                
        if len(group)==1:
            mergeid2event[mergeid] = {
                "event_title": df.loc[group[0], "event_title"],
                "event_description": df.loc[group[0], "event_description"]
            }
            id2mergeid[str(group[0]+accum_count)] = mergeid
            # cluster_merge[str(group)] = df.loc[group[0], "topic"]
            # cluster_merge2[group[0]] = df.loc[group[0], "topic"]
    
    # 更新聚类结果中的 event_id
    _df = pd.read_csv(
        cluster_file_path, 
        encoding="utf-8", 
        keep_default_na=False  # 空值都是默认为 NAN，设置 keep_default_na=False 让读取出来的空值是空字符串
    )
    _df = _df.astype(str)
    _df["event_id"] = _df["event_id"].map(id2mergeid)
    _df.to_csv(cluster_file_path.replace(".csv", "_merge.csv"), index=False, encoding="utf-8")
    print("更新帖子聚簇结果保存在：", cluster_file_path.replace(".csv", "_merge.csv"))

    # 写入csv
    tmp = []
    for k, v in mergeid2event.items():
        tmp.append([k, v["event_title"], v["event_description"]])
    
    _df = pd.DataFrame(tmp, columns=['event_id', 'event_title', 'event_description'])
    _df.to_csv(merge_file_path, index=False, encoding="utf-8")
    print("更新事件结果保存在：", merge_file_path)


if __name__ == "__main__":
    
    # 聚簇合并
    cluster_merge(cluster_file_path, topic_file_path, merge_file_path)

    # # TODO：无关事件过滤，避免干扰其他话题？
    # df = pd.read_csv(merge_file_path, encoding="utf-8", keep_default_na=False)
    # df = df.astype(str)
    # # 如果事件topic跟理想无关，过滤
    # df = df[df.topic.str.contains("|".join(["理想", "蔚小理", "李想", "造车新势力", "增程"]))]
    # df.to_csv(merge_file_path.replace(".csv", "_filter.csv"), index=False)
    # print("聚簇合并过滤结果保存在：", merge_file_path.replace(".csv", "_filter.csv"))
