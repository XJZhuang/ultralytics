"""
    生成包含所有图片路径的 train.txt / val.txt

"""

import os
from typing import List, Tuple


def get_all_images_in_subdir(
    img_subdir: str,  # 图片子文件夹路径（如./images/train01）
    label_subdir: str,  # 对应标签子文件夹路径（如./labels/train01）
    img_extensions: List[str]
) -> Tuple[List[str], int, int]:
    """
    获取单个图片子文件夹中所有图片路径，并统计有/无标签的数量
    :return: (所有图片路径列表, 有标签的图片数, 无标签的图片数)
    """
    # 检查图片子文件夹是否存在
    if not os.path.isdir(img_subdir):
        raise NotADirectoryError(f"图片子文件夹不存在：{img_subdir}")

    # 获取该文件夹下所有符合扩展名的图片
    all_images = []
    for file in os.listdir(img_subdir):
        # 检查文件扩展名是否在支持列表中（忽略大小写）
        file_ext = os.path.splitext(file)[1].lower()
        if file_ext in img_extensions:
            all_images.append(file)

    if not all_images:
        print(f"警告：图片子文件夹 {img_subdir} 中未找到任何图片，已跳过")
        return [], 0, 0

    # 统计有标签和无标签的图片数量
    img_paths = []
    labeled_count = 0
    unlabeled_count = 0

    for img_file in all_images:
        # 图片绝对路径
        img_path = os.path.abspath(os.path.join(img_subdir, img_file))
        img_paths.append(img_path)

        # 检查是否有对应标签（标签文件名=图片文件名，扩展名改为.txt）
        img_base = os.path.splitext(img_file)[0]
        label_file = f"{img_base}.txt"
        label_path = os.path.join(label_subdir, label_file)

        if os.path.isfile(label_path):
            labeled_count += 1
        else:
            unlabeled_count += 1

    return img_paths, labeled_count, unlabeled_count


def generate_train_txt_include_unlabeled(
    img_root: str,
    label_root: str,
    selected_subdirs: List[str],
    output_txt: str,
    img_extensions: List[str] = [".jpg", ".jpeg", ".png", ".bmp", ".gif"]
) -> None:
    """
    生成包含所有图片（有/无标签）路径的train.txt
    :param img_root: 图片根目录（如./images）
    :param label_root: 标签根目录（如./labels）
    :param selected_subdirs: 需要包含的子文件夹（如["train01", "train02"]）
    :param output_txt: 输出文件路径
    :param img_subdir: 子文件夹内图片的相对路径（默认"images"）
    :param label_subdir: 子文件夹内标签的相对路径（默认"labels"）
    :param img_extensions: 支持的图片扩展名
    """
    # 检查根目录是否存在
    if not os.path.isdir(img_root):
        raise NotADirectoryError(f"图片根目录不存在：{img_root}")
    if not os.path.isdir(label_root):
        raise NotADirectoryError(f"标签根目录不存在：{label_root}")

    # 收集所有图片路径
    all_img_paths = []
    total_labeled = 0
    total_unlabeled = 0

    for subdir in selected_subdirs:
        # 构建当前子文件夹的图片路径和标签路径
        img_subdir = os.path.join(img_root, subdir)  # 如./images/train01
        label_subdir = os.path.join(label_root, subdir)  # 如./labels/train01

        print(f"开始处理子文件夹：{subdir}（图片路径：{img_subdir}）")
        try:
            img_paths, labeled, unlabeled = get_all_images_in_subdir(
                img_subdir=img_subdir,
                label_subdir=label_subdir,
                img_extensions=img_extensions
            )
            all_img_paths.extend(img_paths)
            total_labeled += labeled
            total_unlabeled += unlabeled
            print(f"子文件夹 {subdir} 处理完成：总图片数={len(img_paths)}，有标签={labeled}，无标签={unlabeled}")
        except Exception as e:
            print(f"子文件夹 {subdir} 处理失败：{str(e)}，已跳过")

    # 写入输出文件
    if all_img_paths:
        with open(output_txt, "w", encoding="utf-8") as f_out:
            for path in all_img_paths:
                f_out.write(f"{path}\n")
        print(f"\n所有子文件夹处理完成：")
        print(f" - 总图片数：{len(all_img_paths)}")
        print(f" - 有标签的图片数：{total_labeled}")
        print(f" - 无标签的图片数：{total_unlabeled}")
        print(f"图片路径文件已保存至：{os.path.abspath(output_txt)}")
    else:
        print("\n警告：未收集到任何图片路径，未生成输出文件")


if __name__ == "__main__":
    # ---------------------- 配置参数（根据实际路径修改） ----------------------
    IMG_ROOT = r"D:\1_Python\datasets\switchgear\images"  # 图片根目录（包含train01、train02等子文件夹）
    LABEL_ROOT = r"D:\1_Python\datasets\switchgear\labels"  # 标签根目录（包含train01、train02等子文件夹）
    SELECTED_SUBDIRS = ["train01", "train02", "train03", "train04", ]  # 指定需要包含的子文件夹
    # SELECTED_SUBDIRS = ["val01", "val02", "val03", "val04", ]  # 指定需要包含的子文件夹
    OUTPUT_TXT = r"D:\1_Python\datasets\switchgear\labels\train.txt"  # 输出文件路径
    # OUTPUT_TXT = r"D:\1_Python\datasets\switchgear\labels\val.txt"  # 输出文件路径

    # 支持的图片扩展名（根据数据集补充，如添加.webp）
    IMG_EXTENSIONS = [".jpg", ".jpeg", ".png", ".bmp"]
    # -------------------------------------------------------------------------

    try:
        generate_train_txt_include_unlabeled(
            img_root=IMG_ROOT,
            label_root=LABEL_ROOT,
            selected_subdirs=SELECTED_SUBDIRS,
            output_txt=OUTPUT_TXT,
            img_extensions=IMG_EXTENSIONS
        )
    except Exception as e:
        print(f"执行失败：{str(e)}")