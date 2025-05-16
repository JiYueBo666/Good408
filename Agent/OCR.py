import os
import base64
import logging
from openai import OpenAI
from PIL import Image
from dotenv import load_dotenv, find_dotenv

logger=logging.getLogger(__name__)

# 加载环境变量
load_dotenv(find_dotenv())

class OCRModule:
    def __init__(self,openai_api_key,base_url,model_name='gpt-4o') -> None:
        self.api_key=openai_api_key
        self.base_url=base_url
        self.model_name=model_name
        self.init_instance()

    def init_instance(self):
        if self.api_key and self.base_url:
            self.client=OpenAI(api_key=self.api_key,base_url=self.base_url)
        else:
            logger.info("no api key and base url pass to OCR part,use .env file")
            self.api_key=os.getenv('OPENAI_API_KEY')
            self.base_url=os.getenv('OPENAI_API_BASE')
            self.client=OpenAI(api_key=self.api_key,base_url=self.base_url)

    def extract_text(self,image_path):
        """
        使用OpenAI的GPT-4o Vision API从图片中提取文字
        Args:
            image_path: 图片路径
        Returns:
            提取的文字内容
        """
            # 读取图片并转换为base64
        with open(image_path, "rb") as image_file:
            base64_image = base64.b64encode(image_file.read()).decode('utf-8')
            response = self.client.chat.completions.create(
        model=self.model_name,
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "请提取这张图片中的所有文字内容，只返回文字，不需要任何解释或分析。"
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{base64_image}"
                        }
                    }
                ]
            }
        ],
        max_tokens=1000
    )
            # 提取结果
        extracted_text = response.choices[0].message.content
        return extracted_text

    def optimize_image_for_ocr(self,image_path, output_path=None):
        """
        优化图片以提高OCR识别率
        Args:
            image_path: 原图片路径
            output_path: 优化后图片保存路径，如果为None则覆盖原图片
        
        Returns:
            优化后的图片路径
        """
        if output_path is None:
            output_path = image_path
        
        # 打开图片
        img = Image.open(image_path)
        
        # 转换为RGB模式（如果是RGBA或其他模式）
        if img.mode != 'RGB':
            img = img.convert('RGB')
        
        # 调整图片大小（如果太大）
        max_size = 4000  # OpenAI API对图片大小有限制
        if max(img.size) > max_size:
            ratio = max_size / max(img.size)
            new_size = (int(img.size[0] * ratio), int(img.size[1] * ratio))
            img = img.resize(new_size, Image.LANCZOS)
        
        # 保存优化后的图片
        img.save(output_path, 'JPEG', quality=95)
        return output_path

    def extract_text_from_image(self,image_path):
        """
        从图片中提取文字
        Args:
            image_path: 图片路径
        Returns:
            提取的文字内容
        """
        # 优化图片（可选）
        try:
            optimized_image = self.optimize_image_for_ocr(image_path)
            extracted_text = self.extract_text(optimized_image)
            return extracted_text
        except Exception as e:
            print(f"提取文字错误: {e}")
            return None
