#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@Project ：switchgear-ai 
@File    ：api01.py
@Author  ：DELL(zhuangxujun)
@Date    ：2025/10/27 17:24 
@explain : 
'''


from fastapi import FastAPI
from pydantic import BaseModel

# 创建FastAPI应用实例
app = FastAPI()


# 定义GET请求的路由：访问根路径（/）时触发
@app.get("/")
async def read_root():
    # 返回字典（会自动转为JSON响应）
    return {"message": "Hello, FastAPI!"}


# 路径参数：{user_id}是动态值，函数参数需与路径参数同名
@app.get("/users/{user_id}")
async def read_user(user_id: int):  # 类型注解会自动验证（如传入字符串会报错）
    return {"user_id": user_id, "message": "用户信息"}


# 另一个示例：路径参数支持字符串
@app.get("/items/{item_name}")
async def read_item(item_name: str):
    return {"item_name": item_name, "stock": 100}


# 查询参数：skip和limit是可选参数（有默认值）
@app.get("/items/")
async def read_items(skip: int = 0, limit: int = 10):
    # 模拟返回第skip到skip+limit条数据
    return {"skip": skip, "limit": limit, "data": [f"item_{i}" for i in range(skip, skip+limit)]}


# 定义请求体模型（自动验证字段类型和约束）
class Item(BaseModel):
    name: str  # 必选，字符串类型
    price: float  # 必选，浮点型（自动验证正数）
    is_offer: bool | None = None  # 可选，布尔型，默认None


# 定义POST请求，接收Item类型的请求体
@app.post("/items/")
async def create_item(item: Item):  # 参数item会自动解析请求体JSON并验证
    # 处理数据（如存入数据库），这里简单返回
    return {"item_name": item.name, "item_price": item.price, "is_offer": item.is_offer}


