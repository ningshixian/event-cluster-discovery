# export FAQ_ENV=prod     # 线上生产环境
# export AIKOL_DB=da_defeat_act     # 线上生产环境
# conda activate pp

# cd data
# python process.py
# cd ..

# 存量聚类
python a_post_clustering.py 
python b_topic_generation.py 
python c_merge_cluster.py 
python d_new_event_discovery.py

# 新帖聚类
# 记得要清空topic的内容以及存量表数据
nohup python -u f_kafka.py > f.log 2>&1 &
nohup python -u crontab_script.py > crontab.log 2>&1 &

