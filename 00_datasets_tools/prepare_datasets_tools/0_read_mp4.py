# -*- coding: utf-8 -*-
'''
@Project:   ultralytics-main
@File:      0_read_mp4.py
@Author:    123（zhuangxujun）
@Date:      2025/1/23 16:16
@Explain:   读取视频文件并展示，没有其他功能
'''

import os.path
import cv2


def read_videos(video_path, interval):
    # 打开视频文件
    video = cv2.VideoCapture(video_path)
    width = int(video.get(3))
    high = int(video.get(4))
    print(f"video width=={width}, high=={high}, fps=={video.get(5)}")
    success = True

    while success:
        # 读取视频帧
        success, image = video.read()
        if not success:
            print("结束")
            break
        cv2.imshow("read_videos", image)
        if cv2.waitKey(1) & 0xFF == ord('q'):  # Hit 'q' on the keyboard to quit!
            break

    # 释放视频对象
    video.release()


if __name__ == '__main__':
    video_path_1 = r"D:\mp4_videos\20250122\盐田_验电\莲塘"  # 视频文件路径
    video_name = "/20250122084206-20250122084706_ALARM.mp4"
    video_path = video_path_1 + video_name
    interval = 1  # 抽帧间隔，每隔10帧抽取一帧
    read_videos(video_path, interval)

