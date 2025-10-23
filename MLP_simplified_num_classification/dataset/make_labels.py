import argparse
import os
import cv2
import json

def label_images_in_folder(input_dir, output_dir):
    """
    依次读取一个文件夹中的图片，允许用户通过终端输入标签，
    并将标签以JSON格式保存到另一个文件夹中。

    Args:
        input_dir (str): 包含图片的输入文件夹路径。
        output_dir (str): 用于存储生成的JSON文件的输出文件夹路径。
    """
    # 确保输出目录存在，如果不存在则创建
    os.makedirs(output_dir, exist_ok=True)
    print(f"JSON 文件将被保存在: {output_dir}")

    # 支持的图片文件扩展名
    supported_extensions = ('.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.gif')

    # 获取并排序目录中的所有文件，以确保处理顺序一致
    try:
        files = sorted([f for f in os.listdir(input_dir) if f.lower().endswith(supported_extensions)])
    except FileNotFoundError:
        print(f"错误: 输入目录不存在 -> {input_dir}")
        return

    if not files:
        print(f"在目录 '{input_dir}' 中没有找到支持的图片文件。")
        return

    print(f"共找到 {len(files)} 张图片。按 'Enter' 键确认标签并继续，输入 'exit' 退出。")

    for filename in files:
        # 构建完整的文件路径
        img_path = os.path.join(input_dir, filename)

        # 读取图片
        try:
            image = cv2.imread(img_path)
            if image is None:
                print(f"警告: 无法读取图片 {filename}。跳过此文件。")
                continue
        except Exception as e:
            print(f"读取图片 {filename} 时发生错误: {e}")
            continue

        # 显示图片
        window_name = f"Labeling: {filename}"
        cv2.imshow(window_name, image)
        
        # 等待按键事件，给用户时间查看图片
        # 这里的参数 1 是为了让窗口能够响应，否则 input() 会阻塞进程
        cv2.waitKey(1)

        # 在终端提示输入标签
        prompt = f"请输入 '{filename}' 的标签: "
        user_input = input(prompt)

        # 检查是否要退出
        if user_input.lower() == 'exit':
            print("用户请求退出。")
            cv2.destroyAllWindows()
            break

        # 关闭当前图片窗口
        cv2.destroyWindow(window_name)

        # 准备JSON数据
        label_data = {
            "image_filename": filename,
            "label": user_input
        }

        # 构建输出JSON文件的路径
        json_filename = os.path.splitext(filename)[0] + '.json'
        json_path = os.path.join(output_dir, json_filename)

        # 将数据写入JSON文件
        try:
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(label_data, f, ensure_ascii=False, indent=4)
            print(f"-> 标签已保存至 {json_path}")
        except IOError as e:
            print(f"无法写入JSON文件 {json_path}: {e}")

    cv2.destroyAllWindows()
    print("所有图片处理完毕。")

if __name__ == "__main__":
    # 创建命令行参数解析器
    parser = argparse.ArgumentParser(description="一个用于手动标注图片的Python脚本。")

    # 添加 --input_dir 参数
    parser.add_argument(
        "--input_dir",
        type=str,
        required=True,
        help="包含待标注图片的输入文件夹的路径。"
    )

    # 添加 --output_dir 参数
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="用于存储生成的JSON标签文件的输出文件夹的路径。"
    )

    # 解析命令行参数
    args = parser.parse_args()

    # 调用主函数
    label_images_in_folder(args.input_dir, args.output_dir)