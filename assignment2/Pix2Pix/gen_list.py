import os

base_path = 'facades'

def generate_list(mode):
    folder_path = os.path.join(base_path, mode)
    list_file = f'{mode}_list.txt'
    
    if not os.path.exists(folder_path):
        print(f"错误: 找不到文件夹 {folder_path}，请确认数据集已正确解压。")
        return

    # 获取所有 jpg 文件并按名称排序
    files = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith('.jpg')]
    files.sort()

    with open(list_file, 'w') as f:
        for file_path in files:
            # 写入相对路径
            f.write(file_path + '\n')
    
    print(f"成功生成 {list_file}，包含 {len(files)} 张图片。")

if __name__ == '__main__':
    # 生成训练集列表
    generate_list('train')
    # 生成验证集列表
    generate_list('val')
