#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@Project ：switchgear-ai 
@File    ：operate_json.py
@Author  ：DELL(zhuangxujun)
@Date    ：2025/10/22 17:40 
@explain : 
'''

import json
import os
from typing import Dict, List, Any, Set
from pathlib import Path

from ultralytics.utils import LOGGER
from utils.unique_identity import get_uid


class CalibrationFileManager:
    def __init__(self, base_dir: str):
        """初始化标定文件管理器
        :param base_dir: 标定文件存储根目录
        """
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)  # 确保目录存在

    def _get_file_path(self, template_id: str) -> Path:
        """获取标定文件路径（约定文件名格式为 template_{id}.json）"""
        return self.base_dir / f"template_{template_id}.json"

    def _validate_template_structure(self, template_data: Dict, is_new: bool) -> None:
        """基础结构与关键字段校验（逻辑不变）"""
        required_fields = ["template_name", "groups"]
        for field in required_fields:
            if field not in template_data:
                raise ValueError(f"缺少必填字段: {field}")

        # template_name非空校验
        template_name = template_data["template_name"].strip()
        if not template_name:
            raise ValueError("template_name不允许为空")
        template_data["template_name"] = template_name

        # groups结构校验
        if not isinstance(template_data["groups"], list):
            raise ValueError("groups必须是列表类型")
        if not is_new and len(template_data["groups"]) == 0:
            raise ValueError("修改模板时至少保留一个组")

        # is_deleted字段处理
        if "is_deleted" not in template_data:
            template_data["is_deleted"] = False
        elif not isinstance(template_data["is_deleted"], bool):
            raise ValueError("is_deleted必须是布尔值（True/False）")

        # 组与检测框详细校验
        group_names = []  # 校验group_name唯一性
        for group_idx, group in enumerate(template_data["groups"]):
            if not isinstance(group, dict):
                raise ValueError(f"第{group_idx+1}个组必须是字典类型")

            # group_name非空+唯一
            group_name = group.get("group_name", "").strip()
            if not group_name:
                raise ValueError(f"第{group_idx+1}个组的group_name不允许为空")
            if group_name in group_names:
                raise ValueError(f"第{group_idx+1}个组的group_name重复: {group_name}")
            group_names.append(group_name)
            group["group_name"] = group_name

            # detections结构
            if "detections" not in group or not isinstance(group["detections"], list):
                raise ValueError(f"第{group_idx+1}个组缺少detections列表")
            if len(group["detections"]) == 0:
                raise ValueError(f"第{group_idx+1}个组至少保留一个检测框")

            # 检测框字段校验
            messages = []  # 校验当前组内message唯一性
            for det_idx, det in enumerate(group["detections"]):
                if not isinstance(det, dict):
                    raise ValueError(f"第{group_idx+1}组第{det_idx+1}个检测框必须是字典")

                # class_name非空
                class_name = det.get("class_name", "").strip()
                if not class_name:
                    raise ValueError(f"第{group_idx+1}组第{det_idx+1}个检测框class_name为空")
                det["class_name"] = class_name

                # bbox格式
                if "bbox" not in det or not isinstance(det["bbox"], list) or len(det["bbox"]) != 4:
                    raise ValueError(f"第{group_idx+1}组第{det_idx+1}个检测框bbox格式错误")
                try:
                    det["bbox"] = [float(coord) for coord in det["bbox"]]
                except (ValueError, TypeError):
                    raise ValueError(f"第{group_idx+1}组第{det_idx+1}个检测框bbox坐标必须是数字")

                # message非空+唯一
                message = det.get("message", "").strip()
                if not message:
                    raise ValueError(f"第{group_idx+1}组第{det_idx+1}个检测框message为空")
                if message in messages:
                    raise ValueError(f"第{group_idx+1}组第{det_idx+1}个检测框message重复: {message}")
                messages.append(message)
                det["message"] = message

    # 增改
    def save_template(self, template_data: Dict) -> Dict:
        """
        单参数保存模板：
        - template_id为空→新增
        - template_id不为空但模板不存在→自动转为新增
        - template_id不为空且模板存在→修改

        :param template_data: 模板大对象（含template_id、template_name、groups等所有字段）
        :return: 处理后的模板数据（含最终template_id、保存路径等）
        """
        # 1. 预处理template_id（统一为空字符串便于判断）
        input_template_id = template_data.get("template_id", "").strip()
        original_path = self._get_file_path(input_template_id)
        initial_is_new = len(input_template_id) == 0# 初始判断：空→新增，非空→尝试修改

        # 2. 基础结构与关键字段校验（含新增/修改差异化校验）
        # self._validate_template_structure(template_data, is_new)

        # 3. 处理"修改时模板不存在"的情况：转为新增
        final_is_new = initial_is_new
        if not initial_is_new:
            if not original_path.exists():
                # 模板不存在，自动转为新增
                LOGGER.info(f"模板ID {input_template_id} 不存在，自动转为新增操作")
                final_is_new = True  # 标记为新增
                # input_template_id = get_uid()  # 清空无效ID

        # 4. 最终新增逻辑（包含初始新增和转为新增的情况）
        if final_is_new:
            # 3.1 生成全局唯一的template_id
            new_template_id = get_uid()
            template_data["template_id"] = new_template_id  # 回写新ID

            # 3.2 处理组ID和检测框ID（均为新增，生成全局唯一ID）
            for group in template_data["groups"]:
                group["group_id"] = get_uid()  # 组ID全局唯一
                for det in group["detections"]:
                    det["detection_id"] = get_uid()  # 标定框ID全局唯一

            # 3.3 保存新模板文件
            save_path = self._get_file_path(new_template_id)
            with open(save_path, 'w', encoding='utf-8') as f:
                json.dump(template_data, f, ensure_ascii=False, indent=2)
            LOGGER.info(f"新增模板成功，ID: {new_template_id}，保存路径: {save_path}")

        # 5. 最终修改逻辑（仅当模板存在时执行）
        else:
            # 4.2 读取原始模板数据（用于ID对比）
            with open(original_path, 'r', encoding='utf-8') as f:
                original_data = json.load(f)

            # 4.3 提取原始ID映射（组ID→检测框ID集合）
            original_group_ids = {g["group_id"] for g in original_data["groups"]}
            original_group_det_map = {
                g["group_id"]: {det["detection_id"] for det in g["detections"]}
                for g in original_data["groups"]
            }

            # 4.4 处理组ID和检测框ID（新增/修改/删除逻辑）
            processed_groups = []
            for group in template_data["groups"]:
                # 4.4.1 处理组ID（空/不存在→新增；存在→修改）
                input_group_id = group.get("group_id", "").strip()
                if input_group_id in original_group_ids:
                    # 存在→修改：保留原始组ID
                    final_group_id = input_group_id
                else:
                    # 空/不存在→新增：生成新ID
                    final_group_id = get_uid()
                    LOGGER.info(f"组新增，原输入ID: {input_group_id or '空'}，新ID: {final_group_id}")
                group["group_id"] = final_group_id

                # 4.4.2 处理检测框ID（空/不存在→新增；存在→修改）
                original_det_ids = original_group_det_map.get(final_group_id, set())
                for det in group["detections"]:
                    input_det_id = det.get("detection_id", "").strip()
                    if input_det_id in original_det_ids:
                        # 存在→修改：保留原始检测框ID
                        final_det_id = input_det_id
                    else:
                        # 空/不存在→新增：生成新ID
                        final_det_id = get_uid()
                        LOGGER.info(f"组 {final_group_id} 检测框新增，原输入ID: {input_det_id or '空'}，新ID: {final_det_id}")
                    det["detection_id"] = final_det_id

                processed_groups.append(group)

            # 4.5 回写处理后的groups（自动删除未包含的组/检测框）
            template_data["groups"] = processed_groups

            # 4.6 保存修改后的模板（覆盖原文件）
            with open(original_path, 'w', encoding='utf-8') as f:
                json.dump(template_data, f, ensure_ascii=False, indent=2)
            LOGGER.info(f"修改模板成功，ID: {input_template_id}，保存路径: {original_path}")

        # 5. 返回处理后的模板数据（含最终ID、路径等）
        return {
            "success": True,
            "template_data": template_data,
            "save_path": str(self._get_file_path(template_data["template_id"])),
            "operation": "新增" if final_is_new else "修改"
        }

    # 删：逻辑删除（设置is_deleted=true）
    def delete_template(self, template_id: str) -> bool:
        """逻辑删除标定文件
        :param template_id: 模板ID
        :return: 是否删除成功
        """
        data = self.get_template(template_id)
        if data["is_deleted"]:
            return True  # 已删除则直接返回成功

        data["is_deleted"] = True  # 更新删除状态
        file_path = self._get_file_path(template_id)
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        return True

    # 查：查询标定文件
    def get_template(self, template_id: str) -> Dict:
        """查询指定ID的标定文件
        TODO is_deleted为True的不查询
        :param template_id: 模板ID
        :return: 标定文件数据
        """
        file_path = self._get_file_path(template_id)
        if not file_path.exists():
            raise FileNotFoundError(f"标定文件不存在: {template_id}")

        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # self._validate_template_structure(data)
        return data

    def list_templates(self, include_deleted: bool = False) -> List[Dict]:
        """列出所有标定文件（基础信息）
        :param include_deleted: 是否包含已删除的模板
        :return: 模板列表（含ID、名称、状态）
        """
        templates = []
        for file in self.base_dir.glob("template_*.json"):
            try:
                with open(file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                if not include_deleted and data.get("is_deleted", False):
                    continue
                templates.append({
                    "template_id": data["template_id"],
                    "template_name": data.get("template_name", ""),
                    "is_deleted": data.get("is_deleted", False),
                    "group_count": len(data.get("groups", [])),
                    "det_count": sum(len(g.get("detections", [])) for g in data.get("groups", []))
                })
            except Exception as e:
                print(f"读取文件 {file.name} 失败: {str(e)}")
        return templates

    # 增：复制生成新标定文件（自动生成ID）
    def copy_template(self, source_template_id: str, new_name: str = None) -> str:
        """复制已有标定文件生成新文件（自动生成新ID）
        :param source_template_id: 源模板ID
        :param new_name: 新模板名称（可选）
        :return: 新模板ID
        """
        # 读取源文件
        source_data = self.get_template(source_template_id)
        new_template_id = get_uid()  # 生成新ID

        # 复制并修改基础信息
        new_data = {**source_data}  # 浅拷贝
        new_data["template_id"] = new_template_id
        new_data["template_name"] = new_name or f"template_{new_template_id}"
        new_data["is_deleted"] = False  # 新文件默认未删除

        # TODO 同一个templates中，group_name不重复
        # TODO 同一个groups中，detection_name不重复
        # 生成新的detection_id（保留group_id结构，仅更新检测框ID）
        for group in new_data["groups"]:
            # 遍历检测框，生成新ID
            for det in group["detections"]:
                det["detection_id"] = get_uid()

        # 保存新文件
        new_file_path = self._get_file_path(new_template_id)
        with open(new_file_path, 'w', encoding='utf-8') as f:
            json.dump(new_data, f, ensure_ascii=False, indent=2)

        return new_template_id

    # 改：更新标定文件（校验ID不可修改）
    def update_template(self, original_template_id: str, updated_data: Dict) -> None:
        """
        更新模板：非原始ID视为新增，自动生成唯一ID
        :param original_template_id: 被修改的模板文件的id
        :param updated_data: 更新的全量JSON数据
        :return:
        """
        """
        1.对比updated_data中的template_id，禁止修改template_id
        2.template_name不允许为空
        3.group_name不允许重复，不允许为空
        4.同一个detections里，message不允许重复，不允许为空
        5.group_id和detection_id若不存在或为空：保证新的group_name非空且模板内唯一，关键字段非空校验
            （1）为空-->新增，，生成全局id
            （2）非空但不存在-->新增，仍生成全局id，不使用前端的新ID
            （3）非空且存在-->修改。
        6.原来的group_id和detection_id在updated_data不存在，则删除
        """

        # TODO 新增+修改

        # 1. 基础校验与原始数据读取
        self._validate_template_structure(updated_data)
        file_path = self._get_file_path(original_template_id)
        if not file_path.exists():
            raise FileNotFoundError(f"模板不存在: {original_template_id}")

        with open(file_path, 'r', encoding='utf-8') as f:
            original_data = json.load(f)

        # 2. 校验template_id未被修改
        if updated_data["template_id"] != original_template_id:
            raise ValueError("禁止修改template_id")

        # 3. 提取原始数据中的ID（用于对比）
        original_group_ids: Set[str] = {g["group_id"] for g in original_data["groups"]}  # 原始组ID集合
        original_group_det_map: Dict[str, Set[str]] = {  # 原始组ID -> 该组原始检测框ID集合
            g["group_id"]: {det["detection_id"] for det in g["detections"]}
            for g in original_data["groups"]
        }

        # 4. 处理更新数据中的组和检测框（非原始ID视为新增，生成唯一ID）
        processed_groups = []
        used_group_ids = set(original_group_ids)  # 记录当前模板中已使用的组ID（原始+新增），避免重复
        group_names = []  # 记录所有组名（用于校验唯一性）

        for updated_group in updated_data["groups"]:
            # 4.1 处理组ID：非原始ID视为新增，生成唯一ID
            input_group_id = updated_group.get("group_id", "").strip()  # 前端传入的组ID（可能任意值）

            if input_group_id in original_group_ids:
                # 是原始组ID → 保持不变
                group_id = input_group_id
            else:
                # 非原始组ID → 视为新增，生成不重复的ID
                new_group_id = get_uid()
                while new_group_id in used_group_ids:
                    LOGGER.error("get_uid()重复")
                    new_group_id = get_uid()
                group_id = new_group_id
                used_group_ids.add(group_id)  # 加入已使用集合，避免同模板内重复
                LOGGER.info(f"新增组（输入ID: {input_group_id}）→ 生成新ID: {group_id}")

            # 4.2 校验组名（要求3：非空且不重复）
            group_name = updated_group.get("group_name", "").strip()
            if not group_name:
                raise ValueError(f"组ID {group_id} 的group_name不允许为空")
            if group_name in group_names:
                raise ValueError(f"group_name重复: {group_name}（组ID: {group_id}）")
            group_names.append(group_name)

            # 4.3 处理检测框
            processed_detections = []
            # 当前组的原始检测框ID集合（若无则为空）
            original_det_ids = original_group_det_map.get(group_id, set())
            # 记录当前组内已使用的检测框ID（原始+新增），避免重复
            used_det_ids = set(original_det_ids)
            messages = []  # 记录当前组内的message（用于校验唯一性）

            for det in updated_group["detections"]:
                input_det_id = det.get("detection_id", "").strip()  # 前端传入的检测框ID（可能任意值）

                if input_det_id in original_det_ids:
                    # 是当前组的原始检测框ID → 保持不变
                    det_id = input_det_id
                else:
                    # 非当前组原始ID → 视为新增，生成不重复的ID
                    new_det_id = get_uid()
                    while new_det_id in used_det_ids:
                        LOGGER.error("get_uid()重复")
                        new_det_id = get_uid()
                    det_id = new_det_id
                    used_det_ids.add(det_id)  # 加入已使用集合，避免同组内重复
                    LOGGER.info(f"组 {group_id} 新增检测框（输入ID: {input_det_id}）→ 生成新ID: {det_id}")

                # 4.3.2 校验message（要求4：非空且不重复）
                message = det.get("message", "").strip()
                if not message:
                    raise ValueError(f"组 {group_id} 中检测框ID {det_id} 的message不允许为空")
                if message in messages:
                    raise ValueError(f"组 {group_id} 中message重复: {message}（检测框ID: {det_id}）")
                messages.append(message)

                # 保留检测框其他字段，更新ID
                processed_detections.append({**det, "detection_id": det_id, "message": message})  # 确保message是处理后的值

            # 保留组其他字段，更新组ID和名称
            processed_group = {**updated_group,
                               "group_id": group_id,
                               "group_name": group_name,  # 确保组名是处理后的值
                               "detections": processed_detections}
            processed_groups.append(processed_group)

        # 5. 替换更新数据中的groups为处理后的结果
        updated_data["groups"] = processed_groups
        # 确保template_name是处理后的值（去空格）
        updated_data["template_name"] = updated_data["template_name"].strip()

        # 6. 保存更新后的数据
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(updated_data, f, ensure_ascii=False, indent=2)

        LOGGER.info(f"模板 {original_template_id} 更新成功")




# ------------------------------
# 使用示例
# ------------------------------
if __name__ == "__main__":
    # 初始化管理器（标定文件存储在./calibration_json）
    manager = CalibrationFileManager("./calibration_json")

    try:
        # 1. 复制已有模板（假设存在一个template_xxx.json）
        # source_id = "35dcf132e2c743c9b6865e71691c1645"  # 源模板ID
        # new_id = manager.copy_template(source_id, new_name="新模板名称")
        # print(f"复制成功，新模板ID: {new_id}")

        # # 2. 查询模板
        new_id = "8289bc5377504dec8cbe9e244b1fffe3"
        template = manager.get_template(new_id)
        # # print(f"查询到模板: {template['template_name']}")

        # 3. 更新模板（修改group_name和message，保持ID不变）
        template["groups"][0]["group_name"] = "修改后的第一柜"
        template["groups"][0]["detections"][0]["message"] = "修改后的分闸旋钮描述"
        manager.update_template(new_id, template)
        print("模板更新成功")

        # # 4. 逻辑删除模板
        # manager.delete_template(new_id)
        # print("模板标记为已删除")
        #
        # # 5. 列出所有模板
        # all_templates = manager.list_templates()
        # print(f"当前有效模板数量: {len(all_templates)}")

    except Exception as e:
        print(f"操作失败: {str(e)}")
