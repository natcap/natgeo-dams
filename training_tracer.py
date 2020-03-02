"""Tracer code to set up training pipeline."""
import os
import sqlite

import ecoshard
import taskgraph

WORKSPACE_DIR = 'training_tracer_workspace'
CHURN_DIR = os.path.join(WORKSPACE_DIR, 'churn')
ECOSHARD_DIR = os.path.join(WORKSPACE_DIR, 'ecoshard')

KNOWN_DAMS_DATABASE_URL = (
    'https://storage.googleapis.com/natcap-natgeo-dam-ecoshards/'
    'find-the-dams_report_md5_e12d71eb5026751217bd286e9b4c2afe.db')

if __name__ == '__main__':
    for dir_path in [WORKSPACE_DIR, CHURN_DIR, ECOSHARD_DIR]:
        try:
            os.makedirs(dir_path)
        except OSError:
            pass

    task_graph = taskgraph.TaskGraph(CHURN_DIR, 0, 5.0)

    known_dams_database_path = os.path.join(
        ECOSHARD_DIR, os.path.basename(KNOWN_DAMS_DATABASE_URL))
    download_database_task = task_graph.add_task(
        func=ecoshard.download_url,
        args=(KNOWN_DAMS_DATABASE_URL, known_dams_database_path),
        target_path_list=[known_dams_database_path],
        task_name='download %s' % KNOWN_DAMS_DATABASE_URL)


    task_graph.close()
    task_graph.join()
