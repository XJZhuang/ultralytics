#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@Project ：switchgear-ai 
@File    ：82_compare.py
@Author  ：DELL(zhuangxujun)
@Date    ：2025/10/27 9:58 
@explain : 慢
'''

import os
import stat
import hashlib
import paramiko

def calculate_file_hash(file_path, block_size=65536, hash_algorithm='md5'):
    """计算本地文件的哈希值（支持MD5、SHA-256等）"""
    hash_obj = hashlib.new(hash_algorithm)
    with open(file_path, 'rb') as f:
        while chunk := f.read(block_size):
            hash_obj.update(chunk)
    return hash_obj.hexdigest()

def calculate_remote_file_hash(sftp, remote_path, block_size=65536, hash_algorithm='md5'):
    """计算远程文件的哈希值（通过SFTP读取内容）"""
    hash_obj = hashlib.new(hash_algorithm)
    with sftp.file(remote_path, 'rb') as f:
        while chunk := f.read(block_size):
            hash_obj.update(chunk)
    return hash_obj.hexdigest()

def get_local_files_with_hash(local_root, subfolders, img_extensions, hash_algorithm='md5'):
    """收集本地文件及哈希值（{相对路径: 哈希值}）"""
    local_file_hashes = {}
    for subfolder in subfolders:
        folder_path = os.path.join(local_root, subfolder)
        if not os.path.isdir(folder_path):
            print(f"警告：本地文件夹不存在 - {folder_path}")
            continue
        # 递归遍历文件夹
        for root, _, files in os.walk(folder_path):
            for file in files:
                # 检查文件后缀是否为图片
                if any(file.lower().endswith(ext) for ext in img_extensions):
                    file_path = os.path.join(root, file)
                    # 计算相对路径（统一Linux风格）
                    rel_path = os.path.relpath(file_path, local_root).replace(os.sep, '/')
                    # 计算哈希值
                    try:
                        file_hash = calculate_file_hash(file_path, hash_algorithm=hash_algorithm)
                        local_file_hashes[rel_path] = file_hash
                    except Exception as e:
                        print(f"计算本地文件{file_path}哈希失败：{str(e)}")
    return local_file_hashes

def get_remote_files_with_hash(remote_host, remote_port, remote_user, auth, remote_root,
                              subfolders, img_extensions, hash_algorithm='md5'):
    """收集远程文件及哈希值（{相对路径: 哈希值}）"""
    remote_file_hashes = {}
    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())  # 自动接受未知主机密钥

    try:
        # 连接远程服务器（支持密码或密钥认证）
        if isinstance(auth, str):
            # 密码认证
            ssh.connect(remote_host, port=remote_port, username=remote_user, password=auth)
        else:
            # 密钥认证（auth为私钥文件路径）
            private_key = paramiko.RSAKey.from_private_key_file(auth)
            ssh.connect(remote_host, port=remote_port, username=remote_user, pkey=private_key)

        sftp = ssh.open_sftp()  # 打开SFTP会话

        # 递归遍历远程文件夹并计算哈希
        def walk_remote_with_hash(sftp, remote_dir, base_dir):
            file_hashes = {}
            for entry in sftp.listdir_attr(remote_dir):
                entry_name = entry.filename
                entry_path = f"{remote_dir}/{entry_name}"  # Linux路径拼接
                # 判断是否为目录
                if stat.S_ISDIR(entry.st_mode):
                    # 递归处理子文件夹
                    sub_hashes = walk_remote_with_hash(sftp, entry_path, base_dir)
                    file_hashes.update(sub_hashes)
                else:
                    # 检查是否为图片文件
                    if any(entry_name.lower().endswith(ext) for ext in img_extensions):
                        # 计算相对于远程根目录的相对路径
                        rel_path = os.path.relpath(entry_path, base_dir).replace(os.sep, '/')
                        try:
                            # 计算远程文件哈希
                            file_hash = calculate_remote_file_hash(
                                sftp, entry_path, hash_algorithm=hash_algorithm
                            )
                            file_hashes[rel_path] = file_hash
                        except Exception as e:
                            print(f"计算远程文件{entry_path}哈希失败：{str(e)}")
            return file_hashes

        # 遍历所有子文件夹
        for subfolder in subfolders:
            remote_folder = f"{remote_root}/{subfolder}"  # 远程子文件夹路径
            try:
                sftp.stat(remote_folder)
                sub_hashes = walk_remote_with_hash(sftp, remote_folder, remote_root)
                remote_file_hashes.update(sub_hashes)
            except FileNotFoundError:
                print(f"警告：远程文件夹不存在 - {remote_folder}")
            except Exception as e:
                print(f"处理远程文件夹{remote_folder}出错：{str(e)}")

        sftp.close()

    except Exception as e:
        print(f"SSH连接失败：{str(e)}")
        return None
    finally:
        ssh.close()

    return remote_file_hashes

def compare_file_contents(local_hashes, remote_hashes):
    """对比文件内容（基于哈希），返回三种差异"""
    local_paths = set(local_hashes.keys())
    remote_paths = set(remote_hashes.keys())

    # 远程缺失的文件（本地有，远程无）
    missing_in_remote = local_paths - remote_paths
    # 远程多余的文件（远程有，本地无）
    extra_in_remote = remote_paths - local_paths
    # 内容不一致的文件（路径存在但哈希不同）
    common_paths = local_paths & remote_paths
    inconsistent = [path for path in common_paths if local_hashes[path] != remote_hashes[path]]

    return missing_in_remote, extra_in_remote, inconsistent

def save_results(missing, extra, inconsistent, output_file):
    """保存包含内容差异的结果"""
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("=== 1. 远程缺失的文件（本地存在，远程不存在） ===\n")
        for file in sorted(missing):
            f.write(f"{file}\n")

        f.write("\n=== 2. 远程多余的文件（远程存在，本地不存在） ===\n")
        for file in sorted(extra):
            f.write(f"{file}\n")

        f.write("\n=== 3. 内容不一致的文件（路径存在但内容不同） ===\n")
        for file in sorted(inconsistent):
            f.write(f"{file}\n")
    print(f"结果已保存至：{os.path.abspath(output_file)}")


def main():
    # --------------------------
    # 配置参数（请根据实际情况修改）
    # --------------------------
    config = {
        "local_root": r"D:\1_Python\datasets\switchgear\images",  # 本地数据集根目录（Windows路径）
        "remote_host": "10.12.137.31",  # 远程服务器IP
        "remote_port": 53022,  # SSH端口（默认22）
        "remote_user": "root",  # 远程登录用户名
        "auth": "AITest@cet2025",  # 密码（字符串）或私钥路径（如"/home/user/.ssh/id_rsa"）
        "remote_root": "/home/zxj/datasets/switchgear/images",  # 远程数据集根目录（Linux路径）
        "subfolders": ["train01", "train02", "train03", "train04"],  # 需要检查的子文件夹
        "img_extensions": ('.jpg', '.jpeg', '.png', '.bmp'),  # 图片文件后缀
        "hash_algorithm": "md5",  # 哈希算法（可选md5、sha256等）
        "output_file": "81_dataset_content_comparison.txt"  # 结果输出文件
    }

    # 收集本地文件及哈希
    print("正在收集本地文件及计算哈希...")
    local_hashes = get_local_files_with_hash(
        config["local_root"],
        config["subfolders"],
        config["img_extensions"],
        config["hash_algorithm"]
    )
    print(f"本地共处理 {len(local_hashes)} 个图片文件")

    # 收集远程文件及哈希
    print("\n正在收集远程文件及计算哈希...")
    remote_hashes = get_remote_files_with_hash(
        config["remote_host"],
        config["remote_port"],
        config["remote_user"],
        config["auth"],
        config["remote_root"],
        config["subfolders"],
        config["img_extensions"],
        config["hash_algorithm"]
    )
    if not remote_hashes:
        print("无法获取远程文件，程序退出")
        return
    print(f"远程共处理 {len(remote_hashes)} 个图片文件")

    # 对比内容
    print("\n正在对比文件内容...")
    missing, extra, inconsistent = compare_file_contents(local_hashes, remote_hashes)

    # 输出结果
    print(f"\n远程缺失文件：{len(missing)} 个")
    print(f"远程多余文件：{len(extra)} 个")
    print(f"内容不一致文件：{len(inconsistent)} 个")
    save_results(missing, extra, inconsistent, config["output_file"])

if __name__ == "__main__":
    main()