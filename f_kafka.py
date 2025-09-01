import os
os.environ["CUDA_VISIBLE_DEVICES"] = "3"
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import traceback

from kafka import KafkaConsumer
from kafka import KafkaProducer
import requests
import json
import sys
from datetime import datetime, timedelta
import pickle
import numpy as np

import warnings
warnings.filterwarnings("ignore")   # 忽略警告

import sys
sys.path.append("..")
from data.process import clean_post_df

"""
1、从kafka中消费数据，并写入到本地csv文件，方便后续处理。
2、从本地csv文件读取数据，并做数据清洗。
3、使用bert模型对数据进行向量化，并做聚类。
4、读取聚类结果，并做事件发现。
5、将事件发现结果写入到本地csv文件，方便后续处理。
"""
import os

env = os.getenv("AIKOL_ENV", default="test")
database = os.getenv("AIKOL_DB", default="da_defeat_act")
env = "prod"

cluster_file_path = "./dbscan/result/cluster_result_merge.csv"  # 存量帖子聚簇结果
merge_file_path = "./dbscan/result/topic_summary_merge.csv"  # 存量事件
new_data_path = "./data/new_posts.csv"  # 新帖子

# 输出结果文件
path1 = "./dbscan/result/d_cluster_old.csv"
path2 = "./dbscan/result/d_cluster_new.csv"
path3 = "./dbscan/result/d_topic_new.csv"

# group_id = "algo_wubo" # 消费者组
# auto_offset_reset="earliest"
# # 指定group_id：可以让多个消费者协作，每条消息只能被消费1次，实现断点续传；
# # 指定groupid且第一次执行时，auto_offset_reset="earliest"会从最早的数据开始消费。
# # 后续同一个groupid再次执行，则不再读取已消费过的数据，只能消费后续新写入的数据。

group_id = None  # 消费者组
auto_offset_reset = "latest"
# 不指定group_id：则进行广播消费，即任一条消息被多次消费
# auto_offset_reset="earliest"，每次都从最早的数据开始消费,重复消费之前的数据;
# auto_offset_reset="latest"，等待后续新写入时再被消费，不会出现数据缺失的情况。

topic_consumer = "topic_eps_idc_pub_senti_work"  # 要订阅的topic
topic_producer = "topic_eps_idc_pub_senti_algo_event_detail"  # 要订阅的topic，换一个
# topic_producer = 'topic_eps_idc_pub_senti_risk_event_detail'  # 吴博原始topic
# consumer.subscribe(pattern='topic1, topic2, topic3')  # 订阅多个 topic

# 需要设置时间分区？
consumer = KafkaConsumer(
    topic_consumer,  # 要订阅的topic
    bootstrap_servers=[
        ...
    ],  # 消息服务器，需要转移到 12 生产机！！！
    auto_offset_reset=auto_offset_reset,
    group_id=group_id,
    value_deserializer=lambda m: json.loads(m.decode("utf-8")),
)
producer = KafkaProducer(
    bootstrap_servers=[
        ...
    ],  # 消息服务器
    value_serializer=lambda v: json.dumps(v).encode("utf-8"),
)
print("load kafka resource success")

# # 1、确保 Kafka 集群的 auto.create.topics.enable 配置设置为 true
# # 2、创建Kafka主题 https://blog.csdn.net/weixin_44458771/article/details/142493968
# from kafka.admin import KafkaAdminClient, NewTopic
# admin_client = KafkaAdminClient(
#     bootstrap_servers=['172.21.87.116:9092','172.21.87.119:9092','172.21.84.110:9092'],
#     client_id='test')
# topics = admin_client.list_topics()
# print(topics)
# # # 删除主题
# # admin_client.delete_topics(topics=[topic_producer])
# # # 创建主题
# # new_topic = NewTopic(name=topic_producer, num_partitions=3, replication_factor=1)
# # admin_client.create_topics(new_topics=[new_topic], validate_only=False)
# exit()


def publish_to_kafka(data_json, origin_value, new_event_id, event):
    """将指定的数据发布到Kafka。"""
    try:
        # data_json['risk_event_source'] = 'lixiang'
        data_json["event_id"] = new_event_id
        data_json["event_title"] = event["event_title"]  # 事件标题
        data_json["event_description"] = event["event_description"]  # 概要描述
        origin_value["data_json"] = data_json
        producer.send(topic_producer, value=origin_value)
        producer.flush()  # 确保所有消息都被发送
        # print(origin_value)
        # print(datetime.now().strftime("%Y-%m-%d %H:%M:%S"), new_event_id)
    except Exception as e:
        print(f"发送消息时发生错误")
        traceback.print_exc()


def consume_messages():
    """
    从Kafka消费消息，并根据条件处理和保存数据。

    流程：
    初始化用于存储文章信息的字典posts和原始消息的字典msg_ori。
    遍历从consumer接收到的每条消息。
    提取消息中的data_json字段，并从中获取文章的相关信息。
    检查文章的创建时间是否早于设定的时间点（2024年9月25日之前的文章将被忽略）。
    将符合条件的文章信息存入posts字典，并将原始消息存入msg_ori字典。
    当收集到的文章数量达到10000篇时：
    将收集到的文章信息转换为DataFrame对象，并进行数据清洗。
    打印清洗后的帖子数量，并将清洗后的数据保存至CSV文件。
    运行新事件发现模块。
    执行增量更新操作，并将结果发送到Kafka。
    清空posts和msg_ori字典，重置开始时间。
    循环继续处理新的消息。
    """
    posts = {}
    msg_ori = {}
    start_time = datetime.now()
    # start_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    for message in consumer:
        # print(f"Received topic: {message.topic}")
        # print(f"Received key: {message.key}")
        # print(f"Received message: {message.value}")
        msg = message.value
        data_json = msg["data_json"]
        work_title = data_json.get("headline")  # 文章标题
        work_content = data_json.get("content")  # 文章内容
        ocr_content = data_json.get("ocr_content")
        source = data_json.get("source")  # 文章来源
        work_url = data_json.get("doc_url")  # 文章链接
        work_id = data_json.get("doc_id")  # 文章链接
        created_time = data_json.get("created_time")  # 文章发布时间

        # 丢弃created_time时间小于 2024-mm-dd 00:00:00 的数据
        if created_time and datetime.strptime(
            created_time, "%Y-%m-%d %H:%M:%S"
        ) < datetime(
            2024, 11, 12, 13, 18
        ):  # 年月日时分秒
            continue

        posts[work_id] = [
            work_url,
            created_time,
            source,
            work_id,
            work_title,
            work_content,
            ocr_content,
        ]
        msg_ori[work_id] = msg  # 保存原始消息

        # # 如果时间为凌晨的零点
        # if datetime.strptime(created_time, "%Y-%m-%d %H:%M:%S").hour == 0:
        #     return_code = os.system("python d_new_event_discovery_all.py")
        #     if return_code != 0:
        #         print(f"命令执行失败，返回码: {return_code}")
        #     else:
        #         print("命令执行成功")

        # 当消息数量达到 1w 条 or 时间过去 60 分钟
        if len(posts) >= 10000 or (datetime.now() - start_time) >= timedelta(hours=1):
            start_time = datetime.now()
            _posts = pd.DataFrame(
                list(posts.values()),
                columns=[
                    "work_url",
                    "publish_time",
                    "source",
                    "work_id",
                    "work_title",
                    "work_content",
                    "ocr_content",
                ],
            )
            _posts = _posts.astype(str)
            print(datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
            print("一、准备清洗拿到的kafka消息...", _posts.shape)
            _posts = clean_post_df(_posts)  # pass
            print("finish! 写入 new_posts.csv ", _posts.shape)
            _posts.to_csv(new_data_path, index=False, encoding="utf-8")

            if _posts.shape[0] <= 0:
                # 无需执行事件发现，直接发送 kafka 消息
                print(f"无需执行事件发现，直接发送数据 → kafka...({len(msg_ori)})\n\n")
                for work_id, msg in msg_ori.items():
                    publish_to_kafka(
                        msg["data_json"],
                        msg,
                        new_event_id="-1",
                        event={"event_title": "", "event_description": ""},
                    )
            else:  # 有数据才执行新事件发现模块，否则sbert报错
                print("二、执行新事件发现模块...")
                return_code = os.system("python d_new_event_discovery.py")
                if return_code != 0:
                    print(f"命令执行失败，返回码: {return_code}")
                else:
                    print("命令执行成功")

                print("\n三、执行增量更新落表...")
                from e_export import incremental_update_2

                path1 = "./dbscan/result/d_cluster_old.csv"  # 新聚簇结果
                path2 = "./dbscan/result/d_cluster_new_update.csv"  # 新聚簇结果
                path3 = "./dbscan/result/d_topic_new.csv"  # 新增的聚簇对应的事件表
                producer_data = incremental_update_2([path1, path2], path3)
                # df.row = work_id, event_id, event

                # 发送 kafka 消息
                print(f"发送清洗后数据 → kafka...({len(producer_data)})")
                for _, row in producer_data.iterrows():
                    publish_to_kafka(
                        msg_ori[row["work_id"]]["data_json"],
                        msg_ori[row["work_id"]],
                        row["event_id"],
                        row["event"],
                    )

                filtered_msg_ori = {
                    k: v
                    for k, v in msg_ori.items()
                    if k not in producer_data["work_id"].values
                }
                print(f"发送剩余数据 → kafka...({len(filtered_msg_ori)})\n\n")
                for work_id, msg in filtered_msg_ori.items():
                    publish_to_kafka(
                        msg["data_json"],
                        msg,
                        new_event_id="-1",
                        event={"event_title": "", "event_description": ""},
                    )

                posts, msg_ori = {}, {}
                # input()


if __name__ == "__main__":
    consume_messages()
