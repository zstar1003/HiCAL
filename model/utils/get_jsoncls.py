import json

# 读取JSON文件
input_json_path = "../dataset/auair/annotations.json"   # 请将此路径替换为您的JSON文件路径
with open(input_json_path, "r") as file:
    data = json.load(file)

# 输出所有的对象类别名称
categories = data["categories"]
print("对象类别名称列表:")
for category in categories:
    print(category)
