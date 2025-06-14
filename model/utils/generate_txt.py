import os

# 定义文件夹路径
folder_path = '../dataset/auair_part/init/images'  # 替换为你的文件夹路径
output_file = '../dataset/auair_part/init/train.txt'

# 打开输出文件
with open(output_file, 'w') as f:
    # 遍历文件夹中的所有文件
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            # 获取文件名（不带后缀）
            filename_without_extension = os.path.splitext(file)[0]
            # 写入到TXT文件
            f.write(filename_without_extension + '\n')

print(f"所有文件名已写入 {output_file}")
