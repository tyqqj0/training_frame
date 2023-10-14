# -*- CODING: UTF-8 -*-
# @time 2023/10/9 10:46
# @Author tyqqj
# @File stop_runs.py
# @
# @Aim 


from mlflow.tracking import MlflowClient
import mlflow.exceptions


def stop_all_runs():
    # 创建一个 MlflowClient 实例
    client = MlflowClient()
    # 获取所有的运行
    experiment_id = client.get_experiment_by_name("train").experiment_id
    runs = client.search_runs(experiment_ids=[experiment_id])
    print("experiment_id:", experiment_id)
    # 遍历所有的运行
    for run in runs:
        # 如果运行还在进行中，结束它
        try:
            if run.info.status == "RUNNING":
                print("run_id:", run.info.run_id)
                client.set_terminated(run.info.run_id)
        except mlflow.exceptions.MissingConfigException:
            print(f"Skipping run with missing meta.yaml file: {run.info.run_id}")


stop_all_runs()
