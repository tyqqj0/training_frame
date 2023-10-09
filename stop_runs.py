# -*- CODING: UTF-8 -*-
# @time 2023/10/9 10:46
# @Author tyqqj
# @File stop_runs.py
# @
# @Aim 


from mlflow.tracking import MlflowClient


def stop_all_runs():
    # 创建一个 MlflowClient 实例
    client = MlflowClient()
    # 获取所有的运行
    experiment_id = client.get_experiment_by_name("traint1").experiment_id
    runs = client.list_run_infos(experiment_id)
    print("experiment_id:", experiment_id)
    # 遍历所有的运行
    for run in runs:
        # 如果运行还在进行中，结束它
        if run.status == "RUNNING":
            print("run_id:", run.run_id)
            client.set_terminated(run.run_id)


stop_all_runs()
