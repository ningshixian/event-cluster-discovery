import sys
import time
import datetime
import requests
from apscheduler.schedulers.blocking import BlockingScheduler
from apscheduler.triggers.cron import CronTrigger

import subprocess
from pytz import timezone

"""
定时近全量更新 - 定时任务
"""


def scheduled_job():
    # 重启服务
    # loader = subprocess.Popen(["pkill", "-f", "crontab_retraining.py"])
    # returncode = loader.wait()  # 阻塞直至子进程完成
    print("每天凌晨定时增量更新事件，任务启动!")
    print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(time.time())))
    loader = subprocess.Popen(["python", "cron_event_update.py"])
    returncode = loader.wait()  # 阻塞直至子进程完成
    print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(time.time())))
    print("【定时增量更新事件】schedule完成!\n")


def auto_update_json():
    scheduled_job()


def dojob():
    # 创建调度器：BlockingScheduler 
    # 指定时区，例如上海时区
    scheduler = BlockingScheduler(timezone="Asia/Shanghai")

    # 未显式指定，那么则立即执行
    scheduler.add_job(auto_update_json, args=[])

    # # 添加定时任务，每5分钟执行一次
    # scheduler.add_job(scheduled_job, trigger=CronTrigger.from_crontab('*/5 * * * *'), id='event update')

    # 添加定时任务，每天凌晨2点 trigger='cron' 
    scheduler.add_job(scheduled_job, trigger=CronTrigger.from_crontab('0 2 * * *'), id='event update')
    scheduler.start()


if __name__ == "__main__":
    dojob()
