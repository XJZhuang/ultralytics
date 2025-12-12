#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@Project ：ultralytics-main 
@File    ：0_merge_csv.py
@IDE     ：PyCharm 
@Author  ：zhuangxujun
@Date    ：2024/12/2 16:59 
@explain :  合并同一文件夹下的所有视频 指定
（1）区分
（2）班组
（3）日期
'''
import os
import subprocess
import datetime
import argparse

from ultralytics.utils import LOGGER

file_list = []

# 遍历path/dt下的所有文件，加入一个文件夹下有多个视频文件，合并为一个视频文件，并仅将合并后的文件添加到待检测视频列表
def traverse_folder_videos(path, dt, task=None):
    for root, dirs, files in os.walk(path):
        # video_files_in_folder = [os.path.join(root, f) for f in files if is_video_file(f)]
        video_files_in_folder = [os.path.join(root, f) for f in files]
        # 检查当前文件夹的视频文件数量
        if len(video_files_in_folder) >= 2:
            # 按名字升序排序
            video_files_in_folder.sort()
            # 创建合并后的文件名，这里使用第一个视频文件的名字并增加后缀_all os.path.splitext(video_files_in_folder[0])返回元组，文件名和扩展名
            output_path, extension = os.path.splitext(video_files_in_folder[0])
            output_file = output_path + '_all' + extension
            # 合并视频文件
            merge_videos(video_files_in_folder, output_file)
            # 合并后删除文件，节省空间
            # del_videos(video_files_in_folder)
            # 如果合并后的文件路径中包含dt和task（如果task不是None的话），则添加到file_list中
            if dt in output_file and task in output_file:
                file_list.append(output_file)
        else:
            # 检查单独的视频文件是否包含dt和task，并添加到file_list中
            for video in video_files_in_folder:
                if dt in video and task in video:
                    file_list.append(video)


# 创建一个函数，合并视频文件https://blog.51cto.com/u_16213346/11911996
def merge_videos(videos_list, output_file):
    # 确保输入的视频文件存在
    for video in videos_list:
        if not os.path.isfile(video):
            print(f"文件 {video} 不存在！")
            return

    # 创建 FFmpeg 输入格式字符串
    input_str = '|'.join(videos_list)
    # 调用 FFmpeg 执行合并操作
    print(f'执行的命令为-->ffmpeg -i "concat:{input_str}" -codec copy -c:a aac {output_file}')
    subprocess.run(f'ffmpeg -i "concat:{input_str}" -codec copy -c:a aac {output_file}', shell=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')
    dt = datetime.datetime.now().strftime('%Y%m%d')
    parser.add_argument('--date', type=str, default="20250325")
    parser.add_argument('--num', type=int, default=4, help="进程池数量")
    parser.add_argument('--video_path', type=str, default="D://mp4_videos//")
    parser.add_argument('--task', type=str, default="验电", help="任务名称，验电/晨会/转录，对应视频文件夹名称")
    parser.add_argument('--district', type=str, default="坪大", help="区分")
    parser.add_argument('--group', type=str, default="葵涌", help="班组")
    parser.add_argument('--verbose', action='store_true', help="显示日志输出，默认不输出")

    args = parser.parse_args()

    LOGGER.info(f"合并日期为 {args.date}，区分为{args.district}")

    # 获取所有待检测视频的路径
    video_path_per_district = os.path.join(args.video_path, args.date, args.district + '_' + args.task, args.group)
    print(f"合并视频的路径是video_path_per_district-->{video_path_per_district}")
    traverse_folder_videos(video_path_per_district, args.date, task=args.task)

    for file in file_list:
        print(file)

