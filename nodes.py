import os
import json
import base64
import io
import re
import torch
import numpy as np
from PIL import Image

# 尝试导入 dashscope，如果未安装则给出明确错误提示
try:
    import dashscope
    from dashscope import MultiModalConversation
except ImportError:
    raise ImportError(
        "请安装 dashscope 库：pip install dashscope"
    )


class NanGuangImagePromptReverse:
    """
    南光AI图像反推提示词节点
    使用阿里云百炼 DashScope 的多模态模型 Qwen3.5-Plus / Qwen3.6-Plus 分析图像，
    输出中英文双语的提示词。
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "model": (["qwen3.5-plus", "qwen3.6-plus"], {
                    "default": "qwen3.5-plus"
                }),
                "api_key": ("STRING", {
                    "multiline": False,
                    "default": "",
                    "placeholder": "输入 API Key，留空则从环境变量 DASHSCOPE_API_KEY 读取"
                }),
            },
        }

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("prompt_zh", "prompt_en")
    FUNCTION = "reverse_prompt"
    CATEGORY = "南光AI/提示词"
    OUTPUT_NODE = False

    def reverse_prompt(self, image, model, api_key):
        # 处理 API Key
        if not api_key.strip():
            api_key = os.environ.get("DASHSCOPE_API_KEY", "")
        if not api_key:
            raise ValueError(
                "未提供 DashScope API Key。请在节点输入框中填写，"
                "或设置环境变量 DASHSCOPE_API_KEY。"
            )

        # 设置 dashscope 全局 API Key
        dashscope.api_key = api_key

        # 处理批量图像，支持 batch size > 1
        if len(image.shape) == 4:          # [B, H, W, C]
            batch_size = image.shape[0]
            images_list = [image[i] for i in range(batch_size)]
        else:                              # 单张 [H, W, C]
            batch_size = 1
            images_list = [image]

        prompts_zh = []
        prompts_en = []

        for img_tensor in images_list:
            # 将 ComfyUI 的 Tensor 图像转换为 PIL Image
            # 张量值范围通常是 0-1，需要转换为 0-255
            img_np = (img_tensor.cpu().numpy() * 255).astype(np.uint8)
            pil_image = Image.fromarray(img_np)

            # 将 PIL Image 转为 Base64 字符串
            buffered = io.BytesIO()
            pil_image.save(buffered, format="JPEG")
            img_base64 = base64.b64encode(buffered.getvalue()).decode("utf-8")
            img_data_url = f"data:image/jpeg;base64,{img_base64}"

            # 构造多模态请求的消息
            messages = [
                {
                    "role": "system",
                    "content": [
                        {
                            "text": (
                                "你是一个专业的图像分析助手。请仔细观察提供的图像，"
                                "用中文和英文分别生成用于 AI 图像生成的详细提示词。"
                                "要求只返回一个严格的 JSON 对象，格式如下：\n"
                                '{"prompt_zh": "中文提示词", "prompt_en": "English prompt"}'
                            )
                        }
                    ]
                },
                {
                    "role": "user",
                    "content": [
                        {"image": img_data_url},
                        {"text": "请分析这张图片并给出提示词。"}
                    ]
                }
            ]

            # 调用 DashScope 多模态模型
            response = MultiModalConversation.call(
                model=model,
                messages=messages
            )

            # 提取模型输出文本
            if response.status_code != 200:
                raise RuntimeError(
                    f"API 请求失败，状态码: {response.status_code}, "
                    f"错误信息: {response.message}"
                )

            output_text = ""
            try:
                output_text = response.output.choices[0].message.content[0]["text"]
            except Exception as e:
                raise RuntimeError(f"解析 API 返回数据失败: {e}")

            # 解析返回的 JSON（可能被包裹在代码块中）
            json_match = re.search(r'\{.*\}', output_text, re.DOTALL)
            if not json_match:
                raise RuntimeError(f"模型未返回有效的 JSON 格式，返回内容: {output_text}")

            try:
                result = json.loads(json_match.group())
                prompt_zh = result.get("prompt_zh", "")
                prompt_en = result.get("prompt_en", "")
            except json.JSONDecodeError:
                raise RuntimeError(f"无法解析 JSON: {json_match.group()}")

            prompts_zh.append(prompt_zh)
            prompts_en.append(prompt_en)

        # 若 batch_size == 1，直接返回字符串；否则返回列表（ComfyUI 会自动处理）
        if batch_size == 1:
            return (prompts_zh[0], prompts_en[0])
        else:
            return (prompts_zh, prompts_en)


# 节点映射
NODE_CLASS_MAPPINGS = {
    "NanGuangImagePromptReverse": NanGuangImagePromptReverse
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "NanGuangImagePromptReverse": "南光AI图像反推提示词"
}