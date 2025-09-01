import pandas as pd
import traceback
# from utils import sql_cn
import pymysql
from sqlalchemy import create_engine
from tqdm import tqdm

"""
将合并后的聚簇结果导入到数据库
"""

# import argparse
# parser = argparse.ArgumentParser()
# parser.add_argument('-e', '--env', type=str, default='test')
# parser.add_argument('-db', '--database', type=str, default='da_defeat_act')
# args = parser.parse_args()
# database = args.database
# env = args.env

import os
env = os.getenv('AIKOL_ENV', default="test")
database = os.getenv('AIKOL_DB', default="xxx")
env = "prod"

engine = create_engine(
    f'mysql+pymysql://{username}:{password}@{host}:3306/{database}'


class DBTool(object):
    def __init__(self):
        self.conn = None
        self.cursor = None
        self.get_conn()
    
    def __del__(self):
        # 关闭数据库连接
        self.conn.close()

    def get_conn(self):
        try:
            self.conn = pymysql.connect(
                host=host,
                port=3306,
                user=username,
                passwd=password,
                database=database, 
                charset="utf8"
            )
            self.cursor = self.conn.cursor()
            print("数据库连接成功~")
        except Exception as e:
            print("数据库连接失败！")
            traceback.print_exc()
    
    def close(self):
        if self.conn:
            try:
                self.conn.close()
                print("MySQL连接已关闭")
            except Exception as err:
                raise ("MySQL关闭异常: ", str(err))
    
    # 执行查询
    def query(self, sql):
        try:
            self.conn.ping(reconnect=True)
            self.cursor.execute(sql)  # 执行 SQL查询
            result = self.cursor.fetchall()
            return result
        except Exception as e:
            traceback.print_exc()
            raise Exception("sql查询失败：" + str(e))
    
    # 执行非查询类语句
    def ExecNonQuery(self, sql, data):
        # 检查MySQL连接是否还在，不存在的话就重连
        self.conn.ping(reconnect=True)
        flag = False
        try:
            with self.conn.cursor() as cursor:  # 查询游标
                count = cursor.execute(sql, data)
                # print(count)
            self.conn.commit()
            flag = True
        except Exception as err:
            print("SQL语句执行失败时，回滚")
            traceback.print_exc()
            self.conn.rollback()
            exit()
        return flag

    # def ExecNonQueryMany(self, sql, data):
    #     # 检查MySQL连接是否还在，不存在的话就重连
    #     self.conn.ping(reconnect=True)
    #     flag = False
    #     try:
    #         with self.conn.cursor() as cursor:  # 查询游标
    #             cursor.executemany(sql, data)
    #             self.conn.commit()
    #             flag = True
    #             print(f"{cursor.rowcount} 行数据更新成功。")
    #     except Exception as err:
    #         print("SQL语句执行失败时，回滚")
    #         traceback.print_exc()
    #         self.conn.rollback()
    #     return flag


def test_db_tool():
    # sql1 = """INSERT INTO aikol_event_mapping (work_id, event_id) VALUES (%s, %s)"""
    # data1 = ('210b3ae5bdc2b6e191e9e1dbfa76279c', '2')
    # sql2 = """INSERT INTO aikol_event_details (event_id, event_title, event_description) VALUES (%s, %s, %s)"""
    # data2 = ('2', 'test title2', 'test description2')

    # 清空表数据
    sql1 = "delete from aikol_event_mapping"
    data1 = None
    sql2 = "delete from aikol_event_details"
    data2 = None

    mydb = DBTool()
    for sql, data in zip([sql1, sql2], [data1,  data2]):
        mydb.ExecNonQuery(sql, data)

    # result = mydb.query("""
    # SELECT A.work_id, A.event_id, B.event_title, B.event_description
    # FROM aikol_event_mapping A
    # LEFT JOIN aikol_event_details B ON A.event_id = B.event_id
    # """)
    # result = pd.DataFrame(result)
    # result.columns = ["work_id", "event_id", "event_title", "event_description"]
    # result = result.drop_duplicates()   # 去重
    # result = result.sort_values(by=["event_id"])
    # return result


def incremental_update(increment_posts_path, increment_events_path):
    """增量更新（use_fast_mode=False）
    找出存量+增量数据中【新增/更新】的数据，不存在则插入, 存在则更新
    """
    mydb = DBTool()

    # 先拿到存量聚簇数据
    result = mydb.query("""SELECT A.work_id, A.event_id FROM aikol_event_mapping A""")
    result = pd.DataFrame(result, columns=["work_id", "event_id"])
    result = result.drop_duplicates()   # 去重
    result = result.sort_values(by=["event_id"])

    # 读取增量更新后的聚簇结果
    df1 = pd.read_csv(
        increment_posts_path, 
        encoding="utf-8", 
        keep_default_na=False
    )
    df1 = df1[["work_id", "event_id"]].astype(str)
    # 找出增量数据中【新增/更新】的数据
    merged_df = pd.merge(result, df1, how="outer", indicator=True)
    diff_rows = merged_df[merged_df['_merge'] == 'right_only']
    print(diff_rows)
    print("新增聚簇：", diff_rows.shape)    # (570, 3)
    
    # 不存在则插入, 存在则更新
    # ON DUPLICATE KEY UPDATE，如果插入操作发生唯一键冲突，就执行更新操作。
    for index, row in tqdm(diff_rows.iterrows()):
        sql = f"""
        INSERT INTO aikol_event_mapping (`work_id`, `event_id`) VALUES (%s, %s)
        ON DUPLICATE KEY UPDATE 
            `work_id`=VALUES(`work_id`), 
            `event_id`=VALUES(`event_id`);
        """
        data = (str(row["work_id"]), str(row["event_id"]))
        mydb.ExecNonQuery(sql, data)

    # 批量插入新事件落表（事件详情表）
    df_new = pd.read_csv(increment_events_path, encoding="utf-8", keep_default_na=False)
    print("新增事件详情落表：", df_new.shape)
    
    try:
        df_new.to_sql(
            name='aikol_event_details', # table_name
            con=engine,
            index=False,
            # if_exists='replace',  # 覆盖写入
            if_exists='append', # 追加写入
            chunksize=500
        )
    except Exception as e:
        print(e)

    # # 字典映射拿event
    # df_new = {
    #     row["event_id"]: {"event_title": row["event_title"], "event_description": row["event_description"]} 
    #     for i,row in df_new.iterrows()
    # }
    # diff_rows["event"] = diff_rows["event_id"].map(lambda x: df_new[x])
    # return diff_rows


def incremental_update_2(new_post_paths, new_event_path):
    """增量更新（use_fast_mode=True）
    仅针对新帖子的增量，直接附加到存量结果中即可，无需关心新增/更新，简单方便哈哈哈

    ### 主要步骤
    1. **更新事件映射表**:
    - 将新帖子数据追加写入到数据库中的`aikol_event_mapping`表。

    2. **更新事件详情表**:
    - 将新的事件数据追加写入到数据库中的`aikol_event_details`表。

    3. **构造Kafka消息**:
    - 构造从事件ID到对应事件标题和描述的字段映射，并将其作为额外字段添加到帖子数据中。
    - 最终返回构造好的Kafka消息数据。
    """
    post1 = pd.read_csv(new_post_paths[0], encoding="utf-8", keep_default_na=False)
    post2 = pd.read_csv(new_post_paths[1], encoding="utf-8", keep_default_na=False)
    posts = pd.concat([post1, post2], ignore_index=True)
    posts = posts[["work_id", "event_id"]]
    posts = posts.astype(str)
    print("新增帖子落表：", posts.shape)

    try:
        # 数据落表（事件映射表）
        posts.to_sql(
            name='aikol_event_mapping', # table_name
            con=engine,
            index=False,
            # if_exists='replace',  # 覆盖写入
            if_exists='append', # 追加写入
            chunksize=500
        )
    except Exception as e:
        print(e)

    events = pd.read_csv(new_event_path, encoding="utf-8", keep_default_na=False)
    print("新增事件详情落表：", events.shape)

    try:
        # 批量插入新事件落表（事件详情表）
        events.to_sql(
            name='aikol_event_details', # table_name
            con=engine,
            index=False,
            # if_exists='replace',  # 覆盖写入
            if_exists='append', # 追加写入
            chunksize=500
        )
    except Exception as e:
        print(e)

    # 获取事件id与事件详情的映射关系
    merge_file_path = './dbscan/result/topic_summary_merge.csv'     # 存量+增量的事件
    df = pd.read_csv(merge_file_path, encoding="utf-8", keep_default_na=False)
    df = df.astype(str)
    id2event = {
        row["event_id"]: {
            "event_title": row["event_title"], 
            "event_description": row["event_description"]
        } 
        for _, row in df.iterrows()
    }
    id2event["-1"] = {"event_title": "", "event_description": ""}   # TypeError: 'float' object is not subscriptable

    # 构造 kafka 消息
    posts["event"] = posts["event_id"].map(id2event)
    producer_data = posts
    return producer_data


def history_db_import(cluster_file_path, topic_file_path):
    """存量结果导入落表"""
    df1 = pd.read_csv(
        cluster_file_path, 
        encoding="utf-8", 
        keep_default_na=False  # 空值都是默认为 NAN，设置 keep_default_na=False 让读取出来的空值是空字符串
    )
    df1 = df1[["work_id", "event_id"]]
    df1 = df1.astype(str)

    # 读取聚簇&话题结果
    # event_id,event_title,event_description
    df2 = pd.read_csv(
        topic_file_path, 
        encoding="utf-8", 
        keep_default_na=False  # 空值都是默认为 NAN，设置 keep_default_na=False 让读取出来的空值是空字符串
    )
    df2 = df2[["event_id","event_title","event_description"]]
    df2 = df2.astype(str)

    # 数据落表（正式）
    r = df1.to_sql(
        name='aikol_event_mapping', # table_name
        con=engine,
        index=False,
        # if_exists='replace',  # 覆盖写入
        if_exists='append', # 追加写入
        chunksize=500
    )

    r = df2.to_sql(
        name='aikol_event_details', # table_name
        con=engine,
        index=False,
        if_exists='append',
        chunksize=500
    )

    sql_query = """
    SELECT A.work_id, A.event_id, B.event_title, B.event_description
    FROM aikol_event_mapping A
    LEFT JOIN aikol_event_details B ON A.event_id = B.event_id
    """
    result = pd.read_sql(sql_query, engine)
    print(result.tail())
    print("已重新落表", result.shape)


# def query1_file2(cluster_file_path):
#     # 查表拿到聚簇结果，并重新写入文件
#     sql_query = """
#     SELECT A.event_id, A.work_id
#     FROM aikol_event_mapping A
#     """
#     result = pd.read_sql(sql_query, engine)
#     # 新增一列 post 全空
#     result["post"] = ""
#     result.to_csv(cluster_file_path, index=False)


# # 查表拿到聚簇结果，并重新写入文件
# cluster_file_path = './dbscan/result/cluster_result_merge.csv'
# query1_file2(cluster_file_path)
# exit()


if __name__ == "__main__":

    print("清空表数据...")
    test_db_tool()
    
    # 存量结果导入落表 11.02
    cluster_file_path = './dbscan/result/cluster_result_merge.csv'
    topic_file_path = './dbscan/result/topic_summary_merge.csv'
    history_db_import(cluster_file_path, topic_file_path)
    exit()

    # # 增量更新（use_fast_mode=False）
    # increment_posts_path = "./dbscan/result/event_mapping_table.csv"    # 新聚簇结果
    # increment_events_path = './dbscan/result/d_topic_new.csv'           # 新增的聚簇对应的事件表
    # incremental_update(increment_posts_path, increment_events_path)
    
    # 增量更新（use_fast_mode=True）
    path1 = './dbscan/result/d_cluster_old.csv'
    path2 = './dbscan/result/d_cluster_new_update.csv'
    path3 = './dbscan/result/d_topic_new.csv'   # 新增的聚簇对应的事件表
    incremental_update_2([path1, path2], path3)
    
