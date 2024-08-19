import os
from PIL import Image
# 文件路径
input_folder = r"C:\Users\Lzy\OneDrive\桌面\研\MIR\MIR-CycleGAN数据集\MIR-CycleGAN数据集\train"
cloudy_folder = os.path.join(input_folder, 'cloudy_image')
auxiliary_folder = os.path.join(input_folder, 'auxiliary_image')
ground_truth_folder = os.path.join(input_folder, 'ground_truth')

# 创建子文件夹
os.makedirs(cloudy_folder, exist_ok=True)
os.makedirs(auxiliary_folder, exist_ok=True)
os.makedirs(ground_truth_folder, exist_ok=True)

# 获取所有图片文件
images = [f for f in os.listdir(input_folder) if f.endswith('.tif')]
images = [img_file for img_file in images if img_file.split('.')[0].isdigit() and 39001 <= int(img_file.split('.')[0]) <= 42000]

# 处理每一张图片
for img_file in images:
    img_path = os.path.join(input_folder, img_file)

    try:
        # 使用with语句打开图片，这样在处理完图片后会自动关闭文件句柄
        with Image.open(img_path) as img:
            # 检查图片尺寸是否正确
            if img.size != (768, 256):
                print(f"Skipping {img_file}, unexpected size: {img.size}")
                continue

            # 切分图片
            img_cloudy = img.crop((0, 0, 256, 256))
            img_auxiliary = img.crop((256, 0, 512, 256))
            img_ground_truth = img.crop((512, 0, 768, 256))

            # 获取文件名
            img_name = os.path.splitext(img_file)[0]  # 去掉文件扩展名

            # 保存切分后的图片
            img_cloudy.save(os.path.join(cloudy_folder, f"{img_name}.tif"))
            img_auxiliary.save(os.path.join(auxiliary_folder, f"{img_name}.tif"))
            img_ground_truth.save(os.path.join(ground_truth_folder, f"{img_name}.tif"))

        # 打印处理成功的文件
        print(f"Processed and saved: {img_file}")

    except Exception as e:
        # 如果发生错误，打印错误信息
        print(f"Error processing {img_file}: {e}")

print("Image splitting complete!")

