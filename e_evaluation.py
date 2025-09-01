# 聚类效果评估
import numpy as np
import pandas as pd
from sklearn.metrics import silhouette_score    # 轮廓系数
from sklearn.metrics import davies_bouldin_score
from sklearn.metrics import calinski_harabasz_score
from sklearn.metrics import adjusted_rand_score
from sklearn.metrics import normalized_mutual_info_score
from sklearn.metrics import fowlkes_mallows_score
from sklearn.metrics import rand_score
from sklearn.metrics import homogeneity_score
from sklearn.metrics import completeness_score
from sklearn.metrics import v_measure_score
from sklearn.metrics import adjusted_mutual_info_score
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score

"""
聚类效果评估 (准召 acc)
"""

# df = pd.read_csv(
#     "data/wubo_test_post.csv", 
#     encoding="utf-8", 
#     keep_default_na=False, 
#     on_bad_lines='skip',
#     # nrows=30000,
# )
# from sklearn.metrics import classification_report
# print(classification_report(df['label'], df['pred']))     # y_true y_pred
# exit()

data_file_path = './data/test_posts.csv'
cluster_file_path = './dbscan/result/test_cluster.csv'
topic_file_path = './dbscan/result/test_topic_summary.csv'
merge_file_path = './dbscan/result/test_merge.csv'

event2id = {
    "xxx事件": 1,
}

def cluster_evaluation(cluster_file_path):
    pass


def get_embedding_data(file_path):
    df = pd.read_csv(
        file_path, 
        encoding="utf-8", 
        keep_default_na=False, 
        on_bad_lines='skip',
        # nrows=30000,
    )
    post2event = {}
    for i,row in df.iterrows():
        key = row['标题'] + " || " + row['内容']
        post2event[key] = event2id[row['舆情事件']]
    return post2event


post2event = get_embedding_data(data_file_path)
true_label = list(post2event.values())


# 根据 test_merge 的结果，得出 cluster_id 到 event_id 的映射
clusterid2eventid = {
    0: 3, 
    2: 3, 
    1: 2, 
    3: 2, 
    6: 1, 
    7: 4,
    4: -4, 
    5: -5, 
    -1: -1
}

df = pd.read_csv(
    cluster_file_path, 
    encoding="utf-8", 
    keep_default_na=False, 
    on_bad_lines='skip',
    # nrows=30000,
)
post2cluster = {}
for i,row in df.iterrows():
    key = row['query']
    post2cluster[key] = row['cluster_label']

predict_label = []
for k,v in post2event.items():
    post, event_id = k,v
    predict = clusterid2eventid[post2cluster[post]]
    predict_label.append(predict)

# # 找到true_label与predict_label值不同的位置
# for i in range(len(true_label)):
#     if true_label[i] != predict_label[i]:
#         print(true_label[i], predict_label[i])
#         print(list(post2event.keys())[i])
#         input()
    
predict_label2 = [x for x in predict_label if x >= 0]
true_label2 = [x for i, x in enumerate(true_label) if predict_label[i] >= 0]

from sklearn.metrics import classification_report
print(classification_report(true_label, predict_label))     # y_true y_pred
print(classification_report(true_label2, predict_label2))     # y_true y_pred

"""
              precision    recall  f1-score   support

          -5       0.00      0.00      0.00         0
          -4       0.00      0.00      0.00         0
          -1       0.00      0.00      0.00         0
           1       0.95      0.62      0.75        61
           2       0.98      0.87      0.92      2459
           3       0.99      0.84      0.91      2332
           4       1.00      0.52      0.69       830

    accuracy                           0.81      5682
   macro avg       0.56      0.41      0.47      5682
weighted avg       0.99      0.81      0.88      5682


              precision    recall  f1-score   support

           1       0.95      1.00      0.97        38
           2       0.97      0.99      0.98      2122
           3       0.99      0.96      0.98      2044
           4       1.00      1.00      1.00       435

    accuracy                           0.98      4639
   macro avg       0.98      0.99      0.98      4639
weighted avg       0.98      0.98      0.98      4639
"""
