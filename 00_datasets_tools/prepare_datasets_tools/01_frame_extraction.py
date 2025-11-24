# -*- coding: utf-8 -*-
'''
@Project:   ultralytics-main
@File:      1_frame_extraction.py
@Author:    123（zhuangxujun）
@Date:      2025/0/0 8:52
@Explain:   读取视频并抽帧
'''
import os.path

import cv2


def extract_frames(video_path, output_path, interval, group_name, date):
    # 打开视频文件
    video = cv2.VideoCapture(video_path)
    print(f"-->{video_path}")
    width = int(video.get(3))
    high = int(video.get(4))
    print(f"video width=={width}, high=={high}, fps=={video.get(5)}")
    count = 0
    success = True

    while success:
        # 读取视频帧
        success, image = video.read()
        if not success:
            print("结束")
            break

        # 按照指定的间隔抽取帧
        if success and count % interval == 0:
            # 保存帧为图片文件
            # img_name = f"{output_path}/lt_20241128_{int(count/10)}.jpg"
            # img_name = f"{output_path}/stj_20241128_{int(count/10)}.jpg"
            img_name = f"{output_path}/{group_name}_{date}_{int(count/interval)}.jpg"  # ------------------------
            # img_name = f"{output_path}/fy_20241204_{int(count/10)}.jpg"
            if os.path.exists(img_name):
                raise Exception
            cv2.imwrite(img_name, image)
            print(img_name)

        count += 1
        # cv2.imshow("img", image)
        # if cv2.waitKey(1) & 0xFF == ord('q'):  # Hit 'q' on the keyboard to quit!
        #     break

    # 释放视频对象
    video.release()


if __name__ == '__main__':
    group_name = 'fh'                                                           #班组
    # district = r'D:\1_Python\datasets\1_detect_electricity_yantian\images'
    district = r'D:\1_Python\datasets\6_detect_electricity_baoan\images'       #区分
    # district = r'D:\1_Python\datasets\3_detect_electricity_luohu\images'
    # district = r'D:\1_Python\datasets\0_detect_helmet\images'

    # 调用函数进行视频抽帧
    # video_path_1 = r"D:\mp4_videos\20250122\盐田_装维\东湖"  # 视频文件路径
    # video_path_1 = r"D:\mp4_videos\20241218\深汕_验电\鲘门"  # 视频文件路径
    # video_path_1 = r"D:\mp4_videos\20241225\宝安_验电\裕安"  # 视频文件路径
    # video_path_1 = r"D:\mp4_videos\20250122\盐田_验电\盐田港"  # 视频文件路径
    video_path_1 = r"D:\mp4_videos\20250325\宝安_验电\福海"  # 视频文件路径
    # video_path_1 = r"D:\mp4_videos\20241230\罗湖_验电\田贝"  # 视频文件路径
    # video_path_1 = r"D:\mp4_videos\20241225\光明_验电\公明马田"  # 视频文件路径
    # video_path_1 = r"D:\mp4_videos\20241225\龙岗_验电\龙城龙中"  # 视频文件路径
    # video_path_1 = r"D:\mp4_videos\20241225\龙华_验电\民治"  # 视频文件路径
    # video_path_1 = r"D:\mp4_videos\20241218\龙岗_验电\平湖平新"  # 视频文件路径
    video_name = "/20250325080039-20250325080539_ALARM.mp4"
    video_path = video_path_1 + video_name

    output_path = district + r"\train_" + group_name  # 输出图片文件夹路径

    os.makedirs(output_path, exist_ok=True)
    interval = 20  # 抽帧间隔，每隔10帧抽取一帧

    extract_frames(video_path, output_path, interval, group_name, '20250325')

