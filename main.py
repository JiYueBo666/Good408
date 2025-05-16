import os
import tempfile
import streamlit as st
from Agent.OCR import OCRModule
from Agent.TopicParse import StructuredAnalysis


# 设置页面标题
st.set_page_config(page_title="EasyAnswer", layout="wide")

def main():
    # 页面标题
    st.title("📝 EasyAnswer")
    
    # 侧边栏 - 可选配置
    with st.sidebar:
        st.header("配置")
        # 可选：允许用户输入自己的API密钥
        use_custom_api = st.checkbox("使用自定义API配置", False)
        
        if use_custom_api:
            api_key = st.text_input("OpenAI API密钥", type="password")
            base_url = st.text_input("API基础URL", "https://api.openai.com/v1")
            model_name = st.selectbox(
                "选择模型",
                ["gpt-4o", "gpt-4-vision-preview", "gpt-4o-mini"],
                index=0
            )
        else:
            # 使用环境变量中的配置
            api_key = None
            base_url = None
            model_name = "gpt-4o"
    
    # 主界面
    st.subheader("上传图片提取文字")
    
    # 文件上传组件
    uploaded_file = st.file_uploader("选择一张图片", type=["jpg", "jpeg", "png", "bmp", "webp"])
    
    if uploaded_file is not None:
        # 显示上传的图片
        col1, col2 = st.columns(2)
        
        with col1:
            st.image(uploaded_file, caption="上传的图片", use_container_width=True)
        
        # 创建临时文件保存上传的图片
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            temp_path = tmp_file.name
        
        # 提取按钮
        if st.button("提取文字"):
            # 创建一个占位符
            spinner_placeholder = st.empty()
            
            try:
                # 第一个操作：提取文字
                with spinner_placeholder.container():
                    with st.spinner("正在读题，请稍候..."):
                        # 初始化OCR模块
                        ocr = OCRModule(api_key, base_url, model_name)
                        
                        # 提取文字
                        extracted_text = ocr.extract_text_from_image(temp_path)
                
                # 第二个操作：分析题目
                with spinner_placeholder.container():
                    with st.spinner("正在解析题目结构，请稍候..."):
                        unstander = StructuredAnalysis(api_key, base_url, model_name)
                        topic = unstander.analysis(extracted_text)
                
                # 显示分析结果
                st.write(topic)

                # 显示结果
                with col2:
                    st.subheader("提取结果")
                    if extracted_text:
                        st.text_area("提取的文字", extracted_text, height=400)
                        
                        # 提供下载选项
                        st.download_button(
                            label="下载文本文件",
                            data=extracted_text,
                            file_name="extracted_text.txt",
                            mime="text/plain"
                        )
                    else:
                        st.error("未能提取到文字，请尝试其他图片。")
            
            except Exception as e:
                st.error(f"处理过程中出错: {str(e)}")
        
            # 删除临时文件
            try:
                os.unlink(temp_path)
            except:
                pass

if __name__ == "__main__":
    main()


