# SA25001037

import gradio as gr
import cv2
import numpy as np

# Function to convert 2x3 affine matrix to 3x3 for matrix multiplication
def to_3x3(affine_matrix):
    return np.vstack([affine_matrix,[0, 0, 1]])

# Function to apply transformations based on user inputs
def apply_transform(image, scale, rotation, translation_x, translation_y, flip_horizontal):

    # 如果没有上传图片，直接返回None避免报错
    if image is None:
        return None

    # Convert the image from PIL format to a NumPy array
    image = np.array(image)
    # Pad the image to avoid boundary issues
    pad_size = min(image.shape[0], image.shape[1]) // 2
    image_new = np.zeros((pad_size*2+image.shape[0], pad_size*2+image.shape[1], 3), dtype=np.uint8) + np.array((255,255,255), dtype=np.uint8).reshape(1,1,3)
    image_new[pad_size:pad_size+image.shape[0], pad_size:pad_size+image.shape[1]] = image
    image = np.array(image_new)

    ### FILL: Apply Composition Transform 
    # Note: for scale and rotation, implement them around the center of the image （围绕图像中心进行放缩和旋转）
    
    # 1. 获取当前图像的宽高及中心点坐标
    h, w = image.shape[:2]
    cx, cy = w / 2.0, h / 2.0

    # 2. 构建平移矩阵 Translation Matrix
    M_trans = np.array([
        [1, 0, translation_x],[0, 1, translation_y],
        [0, 0, 1]
    ], dtype=np.float32)

    # 3. 构建旋转和缩放矩阵 Rotation & Scale Matrix (围绕中心点)
    # cv2.getRotationMatrix2D 会直接生成包含旋转和缩放的 2x3 矩阵，且已按给定中心点处理
    M_rot_scale_2x3 = cv2.getRotationMatrix2D((cx, cy), rotation, scale)
    M_rot_scale = to_3x3(M_rot_scale_2x3)  # 使用题目给定的辅助函数转为 3x3 矩阵

    # 4. 构建水平翻转矩阵 Flip Matrix (围绕中心点)
    # 如果翻转，x坐标变为 -x + w (这样相当于以cx为轴进行镜像)
    M_flip = np.array([
        [-1 if flip_horizontal else 1, 0, w if flip_horizontal else 0],
        [0, 1, 0],
        [0, 0, 1]
    ], dtype=np.float32)

    # 5. 组合所有变换矩阵 Composite Transform
    # 矩阵乘法的顺序从右到左: 先翻转 -> 再旋转/缩放 -> 最后平移
    M_composite = M_trans @ M_rot_scale @ M_flip

    # 6. 取出最终的 2x3 仿射变换矩阵
    M_affine = M_composite[:2, :]

    # 7. 应用变换，设置 borderValue 为白色以防出现黑色边框，和外面的 pad 颜色保持一致
    transformed_image = cv2.warpAffine(image, M_affine, (w, h), borderValue=(255, 255, 255))

    return transformed_image

# Gradio Interface
def interactive_transform():
    with gr.Blocks() as demo:
        gr.Markdown("## Image Transformation Playground")
        
        # Define the layout
        with gr.Row():
            # Left: Image input and sliders
            with gr.Column():
                image_input = gr.Image(type="pil", label="Upload Image")

                scale = gr.Slider(minimum=0.1, maximum=2.0, step=0.1, value=1.0, label="Scale")
                rotation = gr.Slider(minimum=-180, maximum=180, step=1, value=0, label="Rotation (degrees)")
                translation_x = gr.Slider(minimum=-300, maximum=300, step=10, value=0, label="Translation X")
                translation_y = gr.Slider(minimum=-300, maximum=300, step=10, value=0, label="Translation Y")
                flip_horizontal = gr.Checkbox(label="Flip Horizontal")
            
            # Right: Output image
            image_output = gr.Image(label="Transformed Image")
        
        # Automatically update the output when any slider or checkbox is changed
        inputs =[
            image_input, scale, rotation, 
            translation_x, translation_y, 
            flip_horizontal
        ]

        # Link inputs to the transformation function
        image_input.change(apply_transform, inputs, image_output)
        scale.change(apply_transform, inputs, image_output)
        rotation.change(apply_transform, inputs, image_output)
        translation_x.change(apply_transform, inputs, image_output)
        translation_y.change(apply_transform, inputs, image_output)
        flip_horizontal.change(apply_transform, inputs, image_output)

    return demo

# Launch the Gradio interface
interactive_transform().launch()
