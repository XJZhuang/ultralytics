#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@Project ：switchgear-ai 
@File    ：81_compare_file_name.py
@Author  ：DELL(zhuangxujun)
@Date    ：2025/10/27 10:21 
@explain : 对比本地和远程文件中每个文件是否一一对应，以本地为主，找到远程中多余或缺失的图片。
'''
import os
import stat
import paramiko


def get_local_files(local_root, subfolders, img_extensions):
    """收集本地指定文件夹下的所有图片文件（相对路径）"""
    local_files = set()
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
                    # 计算相对于本地根目录的相对路径（统一用/分隔）
                    rel_path = os.path.relpath(os.path.join(root, file), local_root)
                    rel_path = rel_path.replace(os.sep, '/')  # 转为Linux风格路径
                    local_files.add(rel_path)
    return local_files


def get_remote_files(remote_host, remote_port, remote_user, auth, remote_root, subfolders, img_extensions):
    """收集远程服务器指定文件夹下的所有图片文件（相对路径）"""
    remote_files = set()
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

        # 递归遍历远程文件夹
        def walk_remote(sftp, remote_dir, base_dir):
            files = set()
            for entry in sftp.listdir_attr(remote_dir):
                entry_name = entry.filename
                entry_path = f"{remote_dir}/{entry_name}"  # Linux路径拼接
                # 判断是否为目录
                if stat.S_ISDIR(entry.st_mode):
                    files.update(walk_remote(sftp, entry_path, base_dir))
                else:
                    # 检查是否为图片文件
                    if any(entry_name.lower().endswith(ext) for ext in img_extensions):
                        # 计算相对于远程根目录的相对路径
                        rel_path = os.path.relpath(entry_path, base_dir).replace(os.sep, '/')
                        files.add(rel_path)
            return files

        # 遍历所有子文件夹
        for subfolder in subfolders:
            remote_folder = f"{remote_root}/{subfolder}"  # 远程子文件夹路径
            try:
                sftp.stat(remote_folder)  # 检查文件夹是否存在
                sub_files = walk_remote(sftp, remote_folder, remote_root)
                remote_files.update(sub_files)
            except FileNotFoundError:
                print(f"警告：远程文件夹不存在 - {remote_folder}")
            except Exception as e:
                print(f"处理远程文件夹{remote_folder}出错：{str(e)}")

        sftp.close()

    except Exception as e:
        print(f"SSH连接失败：{str(e)}")
        return None
    finally:
        ssh.close()  # 确保连接关闭

    return remote_files


def compare_files(local_files, remote_files):
    """对比本地和远程文件，返回差异"""
    missing_in_remote = local_files - remote_files  # 远程缺失
    extra_in_remote = remote_files - local_files  # 远程多余
    return missing_in_remote, extra_in_remote


def save_results(missing, extra, output_file):
    """保存对比结果到文件"""
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("=== 远程缺失的文件（本地存在，远程不存在） ===\n")
        for file in sorted(missing):
            f.write(f"{file}\n")

        f.write("\n=== 远程多余的文件（远程存在，本地不存在） ===\n")
        for file in sorted(extra):
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
        # "subfolders": ["train01", "train02", "train03", "train04", "train05"],  # 需要检查的子文件夹
        "subfolders": ["val01", "val02", "val03", "val04", "val05"],  # 需要检查的子文件夹
        "img_extensions": ('.jpg', '.jpeg', '.png', '.bmp'),  # 图片文件后缀
        "output_file": "81_dataset_comparison.txt"  # 结果输出文件
    }

    # 收集本地文件
    print("正在收集本地文件...")
    local_files = get_local_files(
        config["local_root"],
        config["subfolders"],
        config["img_extensions"]
    )
    print(f"本地共发现 {len(local_files)} 个图片文件")

    # 收集远程文件
    print("\n正在收集远程文件...")
    remote_files = get_remote_files(
        config["remote_host"],
        config["remote_port"],
        config["remote_user"],
        config["auth"],
        config["remote_root"],
        config["subfolders"],
        config["img_extensions"]
    )
    if not remote_files:
        print("无法获取远程文件，程序退出")
        return
    print(f"远程共发现 {len(remote_files)} 个图片文件")

    # 对比文件
    print("\n正在对比文件...")
    missing, extra = compare_files(local_files, remote_files)

    # 输出结果
    print(f"\n远程缺失文件：{len(missing)} 个")
    print(f"远程多余文件：{len(extra)} 个")
    save_results(missing, extra, config["output_file"])


if __name__ == "__main__":
    main()