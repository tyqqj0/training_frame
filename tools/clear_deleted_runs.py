# -*- CODING: UTF-8 -*-
# @time 2023/12/5 15:20
# @Author tyqqj
# @File clear_deleted_runs.py
# @
# @Aim 

import os
# import mlflow

from mlflow.tracking import MlflowClient


def main():
    client = MlflowClient()

    # 获取 "已删除" 的运行
    deleted_runs = client.list_deleted_runs()

    # 彻底删除每个运行
    for run in deleted_runs:
        print("run_name: {}, run_id: {}".format(run.info.name, run.info.run_id))
        if input("Delete run (y/n): ") != "y":
            continue
        client.delete_run(run.info.run_id)


if __name__ == '__main__':
    main()
