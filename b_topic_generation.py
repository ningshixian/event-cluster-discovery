import os
os.environ["CUDA_VISIBLE_DEVICES"] = "3"
import torch
import pandas as pd
from collections import defaultdict
import traceback
import sys
sys.path.append("..")
from common import constants
from transformers import AutoTokenizer, AutoModel
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn import metrics

from langchain.output_parsers import PydanticOutputParser
from langchain.output_parsers import StructuredOutputParser, ResponseSchema
from langchain.prompts import (
    PromptTemplate,
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
)
from typing import List
from pydantic import BaseModel, Field, validator

from textrank4zh import TextRank4Keyword, TextRank4Sentence
# import jionlp as jio
from gpt4o_api import generate
import pandas as pd
from tqdm import tqdm

"""2、聚簇话题描述生成
https://bitdm.github.io/2020/projects/P10/final.pdf

遍历 DBSCAN 的聚类结果（除了-1）：
- 1. 将一个聚簇内的所有帖子标题拼接在一起（标题+正文速度太慢），形成一个长文本
- 2. 使用TextRank算法提取长文本中的关键短语(num=20)和关键句子(num=10)，分别作为主题关键词和主题中心句
-   若候选描述句中包含聚簇的实体词，则优先选择该该句作为描述句。
-   若候选句子过长，可对描述句进行句法分析，抽取出其主成分
- 3. 生成事件描述，借助 PE+GPT4o 模型生成各个聚簇的【事件标题+概要描述】
- 4. 结果输出：借助 LangChain 和 pydantic，从模型输出提取和解析结构化数据
"""

# data_file_path = './data/test_posts.csv'
# cluster_file_path = './dbscan/result/test_cluster.csv'
# topic_file_path = './dbscan/result/test_topic_summary.csv'
# merge_file_path = './dbscan/result/test_merge.csv'

data_file_path = './data/posts.csv'
cluster_file_path = './dbscan/result/cluster_result.csv'
topic_file_path = './dbscan/result/topic_summary.csv'
merge_file_path = './dbscan/result/topic_summary_merge.csv'

tr4w = TextRank4Keyword()
tr4s = TextRank4Sentence()


class schema(BaseModel):
    Event_Title: str = Field(description="An event title summarized based on the post cluster")
    Summary_Description: str = Field(description="A brief description of the post cluster content")


def get_prompt_and_parser(prompt_path):
    # 初始化解析器
    output_parser = PydanticOutputParser(pydantic_object=schema)

    # 生成的格式提示符
    format_instructions = output_parser.get_format_instructions()
    # Unicode字符时出现中文乱码 https://www.cnblogs.com/Red-Sun/p/17219414.html
    format_instructions = format_instructions.encode().decode('unicode_escape')
    # print(format_instructions)

    # 加入至template中
    with open(prompt_path, "r", encoding="utf-8") as f:
        template = f.read()

    # 将我们的格式描述嵌入到prompt中去，告诉llm我们需要他输出什么样格式的内容
    prompt = PromptTemplate(
        input_variables=["key_sentences", "keyphrases"],
        partial_variables={"format_instructions": format_instructions},
        template=template
    )
    return prompt, output_parser


def cluster_topic_generate(cluster_file_path):
    """
    根据 DBSCAN 聚类结果，提取并生成各个聚类的主题摘要。

    参数：
    - cluster_file_path (str)：指向包含聚类信息的CSV文件的路径。
    返回值：
    - id2summary (dict)：键为聚类ID，值为一个字典，包含了该聚类的主题标题和概要描述

    功能:
    遍历每个事件ID（除了-1）下的所有记录，提取每个聚类的标题和内容。
    - 首先将每个聚类中的「标题」合并成一个长文本。
    - 关键词提取：提取每个聚类的关键短语。
    - 关键句抽取：生成每个聚类的摘要，即挑选出最重要的几句(num=10)来代表整个聚类的主要内容。
    - 生成摘要：借助 GPT4o，根据提取的关键句和关键短语，生成事件标题和概要描述。
    - 结果输出：借助 LangChain 和 pydantic，从模型输出提取和解析结构化数据
    """
    id2summary = {}
    prompt, output_parser = get_prompt_and_parser("prompt.txt")  # 获取 Prompt 和 解析器

    wf = open("./dbscan/result/cluster_key_sentences.txt", 'w') # 存储每个聚簇的关键句
    df = pd.read_csv(cluster_file_path, encoding="utf-8", keep_default_na=False)
    df = df.astype(str)
    df['event_id'] = df['event_id'].astype(int)
    df = df.groupby('event_id')
    for i,content in tqdm(df, total=len(df)):
        cluster_id = int(content["event_id"].values[0])
        print(cluster_id, len(content['post'].values))
        if cluster_id == -1: continue
        cluster_title = "\n".join([x.split(" || ")[0] for x in content['post'].values[:1000]])    # '\n'作为句子切分标识
        # cluster_title = "\n".join([x for x in content['post'].values[:500]])    # 标题+正文（TextRank耗时太长了，截取簇内前 500 个帖子？）

        # 得到每个聚类中的主题关键词
        tr4w.analyze(
            text=cluster_title, 
            lower=True, 
            window=5,   # 窗口大小，int，用来构造单词之间的边。默认值为2。
            vertex_source = 'all_filters',  # 构造pagerank对应的图中的节点
            edge_source = 'no_stop_words',  # 构造pagerank对应的图中的节点之间的边
        )
        # print("/".join(tr4w.words_no_stop_words))
        # print(tr4w.get_keywords(num = 10, word_min_len = 2))
        # 获取 keywords_num 个关键词构造的可能出现的短语，要求这个短语在原文本中至少出现的次数为min_occur_num
        keyphrases = tr4w.get_keyphrases(keywords_num=20, min_occur_num= 2)

        # 得到每个聚类中的抽取式摘要（耗时比较长，再优化下）
        tr4s.analyze(
            text=cluster_title, 
            lower=True, 
            source = 'no_stop_words'
        )
        key_sentences = []
        # 获取最重要的num个长度大于等于sentence_min_len的句子用来生成摘要。
        for item in tr4s.get_key_sentences(num=10, sentence_min_len=6):
            # wf.write(f"{item.index}\t{item.weight}\t{item.sentence}\n")
            key_sentences.append(item.sentence)
        
        # print ("【主题索引】:{} \n【主题帖子量】：{} \n【主题关键词】： {} \n【主题中心句】 ：\n{}".format(cluster_id, len(content["event_id"].values), ','.join([phrase for phrase in keyphrases]), '\n'.join([sen for sen in key_sentences]) ))
        # print ("-------------------------------------------------------------------------")   
        wf.write("【主题索引】:{} \n【主题帖子量】：{} \n【主题关键词】： {} \n【主题中心句】 ：\n{}".format(cluster_id, len(content["event_id"].values), ','.join([phrase for phrase in keyphrases]), '\n'.join([sen for sen in key_sentences]) ))
        # wf.write("\n-------------------------------------------------------------------------\n")   


        # 借助 GPT4o 模型生成话题摘要
        key_sentences = list(set(key_sentences))
        key_sentences = "\n".join([f"{i+1}、{ab}"for i,ab in enumerate(key_sentences)])  # 给每个元素添加编号
        user_prompt = prompt.format(key_sentences=key_sentences, keyphrases=keyphrases)
        # print(user_prompt)
        llm_output, cnt = None, 0
        while not llm_output:
            cnt += 1
            if cnt > 6:
                print("GPT4o 生成失败，请检查输入参数是否正确！")
                return
            try:
                llm_output = generate(user_prompt)
                llm_output = output_parser.parse(llm_output)    # 使用解析器进行解析生成的内容
            except Exception as e:
                print(traceback.format_exc())

        topic_summary = f"事件标题：{llm_output.Event_Title}\n概要描述：{llm_output.Summary_Description}"
        id2summary[cluster_id] = {
            "event_title": llm_output.Event_Title,
            "event_description": llm_output.Summary_Description
        }
        # print (f"\n【主题标题&概要】:{topic_summary}")
        # print ("\n-------------------------------------------------------------------------\n")   
        wf.write(f"\n【主题标题&概要】:{topic_summary}")
        wf.write("\n-------------------------------------------------------------------------\n")   

    wf.close()
    return id2summary


if __name__ == "__main__":

    # 聚簇话题描述生成
    id2summary = cluster_topic_generate(cluster_file_path)

    tmp = []
    for k, v in id2summary.items():
        tmp.append([k, v["event_title"], v["event_description"]])
    
    df = pd.DataFrame(tmp, columns=['event_id', 'event_title', 'event_description'])
    df.to_csv(topic_file_path, index=False)
    print("聚簇话题描述生成结果保存在：", topic_file_path)

