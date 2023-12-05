# -*- CODING: UTF-8 -*-
# @time 2023/12/5 15:20
# @Author tyqqj
# @File clear_deleted_runs.py
# @
# @Aim 

import os
# import mlflow
import shutil

from mlflow.tracking import MlflowClient
from mlflow.entities import ViewType
from mlflow.store.artifact import local_artifact_repo


def main():
    client = MlflowClient(tracking_uri="../mlruns")
    exp = client.get_experiment_by_name("train")
    print("exp:", exp)
    # 获取 "已删除" 的运行
    deleted_runs = client.search_runs(experiment_ids=exp.experiment_id, run_view_type=ViewType.ALL)

    # 彻底删除每个运行
    for run in deleted_runs:
        print(run.info.lifecycle_stage)
        if run.info.lifecycle_stage == "deleted":
            run_name = run.data.tags.get('mlflow.runName')  # 获取运行的名称
            print("run_name: {}, run_id: {}".format(run_name, run.info.run_id))
            # print("run_name: {}, run_id: {}".format(run.info.name, run.info.run_id))
            if input("Delete run (y/n): ") != "y":
                continue
            # 删除运行的所有工件
            # 替换 "mlflow-artifacts:" 前缀
            artifact_uri = run.info.artifact_uri.replace("mlflow-artifacts:", "../mlruns")
            artifact_uri = artifact_uri.replace("artifacts", "")
            # 删除运行的所有工件
            if os.path.exists(artifact_uri):
                shutil.rmtree(artifact_uri)
            else:
                raise ValueError(artifact_uri)
            # client.delete_run(run.info.run_id)
            # print("deleted ", run_name)


def clean_up_artifacts(experiment_name, mlruns_path, mlartifacts_path):
    client = MlflowClient(tracking_uri=mlruns_path)
    exp = client.get_experiment_by_name(experiment_name)
    mlartifacts_path = os.path.join(mlartifacts_path, exp.experiment_id)
    print("exp:", exp)
    # 获取所有的运行
    all_runs = client.search_runs(experiment_ids=[exp.experiment_id], run_view_type=ViewType.ALL)
    all_run_ids = {run.info.run_id for run in all_runs}  # 创建一个包含所有运行 ID 的集合

    # 遍历工件文件夹
    for artifact_dir_name in os.listdir(mlartifacts_path):
        # 检查工件对应的运行是否还存在
        if artifact_dir_name not in all_run_ids:
            print("Artifact directory: {}, run_id: {}".format(artifact_dir_name, artifact_dir_name))
            if input("Delete artifact directory (y/n): ") != "y":
                continue
            # 删除工件文件夹
            artifact_dir_path = os.path.join(mlartifacts_path, artifact_dir_name)
            shutil.rmtree(artifact_dir_path)
            print("deleted ", artifact_dir_path)

if __name__ == '__main__':
    main()
    clean_up_artifacts('debug', '../mlruns', '..\\mlartifacts')
