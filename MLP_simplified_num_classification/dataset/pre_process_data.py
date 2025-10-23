# -- By GeMiNi and Ender_F_L -- #

import os
import argparse
import cv2
from math import floor, log10

def process_images_for_ml(input_dir, output_dir, size):
    """
    使用 OpenCV 批量处理图片，为机器学习项目准备数据集。
    - 调整尺寸
    - 转换为3通道BGR图像
    - 用零填充的数字重命名

    :param input_dir: 包含原始图片的目录路径。
    :param output_dir: 保存处理后图片的目录路径。
    :param size: 一个元组 (width, height)，表示新的图片尺寸。
    """
    # 1. 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)

    # 2. 定义支持的图片格式
    supported_formats = ('.png', '.jpg', '.jpeg', '.bmp', '.tiff')

    # 3. 获取所有有效的图片文件名
    try:
        filenames = [f for f in os.listdir(input_dir) if f.lower().endswith(supported_formats)]
    except FileNotFoundError:
        print(f"错误：输入目录 '{input_dir}' 不存在。请检查路径。")
        return
        
    if not filenames:
        print(f"警告：在 '{input_dir}' 中没有找到任何支持的图片文件。")
        return

    # 4. 准备重命名和打印进度
    counter = 1
    total_files = len(filenames)
    # 计算文件名需要填充的位数，例如120张图就需要3位(001, 002, ..., 120)
    zfill_width = floor(log10(total_files)) + 1
    
    print(f"共找到 {total_files} 个图片文件。开始处理...")

    # 5. 遍历并处理每个文件
    for filename in filenames:
        input_path = os.path.join(input_dir, filename)

        try:
            # 关键点1: 使用 cv2.IMREAD_COLOR 读取图片
            # 这会忽略Alpha透明通道，并将灰度图自动转为3通道BGR图像
            # 这是大多数视觉模型所期望的输入格式
            img = cv2.imread(input_path, cv2.IMREAD_COLOR)

            if img is None:
                print(f"警告: 无法读取或文件已损坏: {filename}。已跳过。")
                continue
            
            # 关键点2: 调整尺寸
            # cv2.INTER_AREA 对于缩小图像能提供高质量的结果，避免摩尔纹
            resized_img = cv2.resize(img, size, interpolation=cv2.INTER_AREA)

            # 关键点3: 创建新的、零填充的文件名
            _, extension = os.path.splitext(filename)
            # 使用 zfill 进行零填充，确保文件按数字顺序正确排序
            new_filename = f"{str(counter).zfill(zfill_width)}{extension}"
            output_path = os.path.join(output_dir, new_filename)

            # 保存处理后的图片
            cv2.imwrite(output_path, resized_img)

            # 打印进度
            print(f"({counter}/{total_files}) 处理完成: {filename} -> {new_filename}")
            counter += 1

        except Exception as e:
            print(f"处理文件 {filename} 时发生未知错误: {e}。已跳过。")

    print("\n所有图片处理完毕！")
    print(f"数据集已准备就绪，保存于: {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="[ML版] 使用OpenCV为机器学习项目准备图片数据集。")
    parser.add_argument('-i', '--input_dir', type=str, required=True, help='原始图片所在的目录。')
    parser.add_argument('-o', '--output_dir', type=str, required=True, help='保存处理后图片的目录。')
    parser.add_argument('-s', '--size', type=int, nargs=2, required=True, metavar=('WIDTH', 'HEIGHT'), help='目标图片尺寸 (宽度 高度)，例如: 224 224')

    args = parser.parse_args()
    target_size = tuple(args.size)

    process_images_for_ml(args.input_dir, args.output_dir, target_size)
