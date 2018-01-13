import pandas as pd
import numpy as np


# 输入原始数据
def split_by_tool(train, test):
    pass


def split_by_tool(data):
    pipeline_dict = get_pipeline_dic_by_tool(data)
    tools = list(pipeline_dict.keys())
    pipelines = []
    for tool in tools:
        pipelines.append(data.loc[:, pipeline_dict[tool]])
    return pipelines
    

def get_pipeline_dic_by_tool(data):
    columns = data.columns
    pipeline_dict = {}
    tool_name = None
    tool_pipeline = []
    for item in columns:
        if item.startswith('t') or item.startswith('T'):
            if len(tool_pipeline) > 0:
                pipeline_dict[tool_name] = tool_pipeline
            tool_name = item
            tool_pipeline = [tool_name]
        else:
            tool_pipeline.append(item)
    return pipeline_dict
