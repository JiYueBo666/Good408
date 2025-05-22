import streamlit as st
from streamlit_chat import message
from PIL import Image
from Agent.OCR import OCRModule
from Agent.TopicParse import StructuredAnalysis, TopicUnderstand
import os
import logging
import asyncio
import nest_asyncio

nest_asyncio.apply()

logging.basicConfig(
    level=logging.DEBUG, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)
logging.getLogger("openai").setLevel(logging.ERROR)
logging.getLogger("httpx").setLevel(logging.ERROR)
logging.getLogger("httpcore").setLevel(logging.WARNING)

# 设置页面配置
st.set_page_config(layout="wide")

# 设置页面标题
st.title("Good408")

# 初始化session state
if "messages" not in st.session_state:
    st.session_state.messages = []
if "processing" not in st.session_state:
    st.session_state.processing = False
if "current_response" not in st.session_state:
    st.session_state.current_response = ""

# 初始化OCR模块和结构化分析模块
ocr_module = OCRModule(openai_api_key=None, base_url=None)  # 使用环境变量中的配置
structured_analysis = StructuredAnalysis(
    openai_api_key=None, base_url=None
)  # 使用环境变量中的配置

TopicUnderstander = TopicUnderstand()

# 创建三列布局：对话区(5) - 图片上传区(3) - 分析结果区(4)
col1, col2, col3 = st.columns([5, 3, 4])

with col1:
    st.subheader("对话区")
    # 显示聊天历史
    chat_container = st.container()
    with chat_container:
        for i, msg in enumerate(st.session_state.messages):
            if msg["role"] == "user":
                message(msg["content"], is_user=True, key=f"user_{i}")
            else:
                message(msg["content"], is_user=False, key=f"assistant_{i}")

        # 显示正在生成的回复
        if st.session_state.processing:
            with st.empty():
                message(st.session_state.current_response, is_user=False, key="current")

    # 用户输入框固定在底部
    st.text_input(
        "请输入您的问题：", key="user_input", on_change=lambda: handle_user_input()
    )

with col2:
    st.subheader("图片上传")
    uploaded_file = st.file_uploader("选择一张图片", type=["png", "jpg", "jpeg"])

    if uploaded_file is not None:
        # 显示上传的图片
        image = Image.open(uploaded_file)

        # 创建临时文件路径
        temp_path = os.path.join("temp", uploaded_file.name)
        os.makedirs("temp", exist_ok=True)

        # 保存上传的文件
        with open(temp_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        # 添加OCR处理按钮
        if st.button("开始处理"):
            st.session_state.processing = True
            progress_bar = st.progress(0)
            status_text = st.empty()

            # OCR处理
            status_text.text("正在进行OCR识别...")
            progress_bar.progress(25)
            optimized_path = ocr_module.optimize_image_for_ocr(temp_path)
            progress_bar.progress(50)
            text = ocr_module.extract_text(optimized_path)
            progress_bar.progress(75)

            # 结构化分析
            status_text.text("正在进行结构化分析...")
            analysis_result = structured_analysis.analysis(text)
            logger.info(f"结构化解析结果{analysis_result}")

            if "error" not in analysis_result:
                result = TopicUnderstander.analysis(
                    content=text, struct=analysis_result
                )
                logger.info(f"AI 分析结果 {result}")

                # 更新session state
                st.session_state.messages.append({"role": "user", "content": text})
                st.session_state.messages.append(
                    {"role": "assistant", "content": result["analysis"]}
                )

                # 更新分析结果
                st.session_state.current_struct = result["struct"]
                st.session_state.current_more = result["more"]
            else:
                st.error("结构化分析失败")

            progress_bar.progress(100)
            status_text.text("处理完成！")
            st.session_state.processing = False
            st.rerun()

with col3:
    st.subheader("分析结果")

    # 结构内容
    struct_container = st.container()
    with struct_container:
        st.markdown("### 结构")
        if "current_struct" in st.session_state:
            st.write(st.session_state.current_struct)

    # 拓展内容
    more_container = st.container()
    with more_container:
        st.markdown("### 拓展")
        if "current_more" in st.session_state:
            st.write(st.session_state.current_more)


# 处理用户输入的函数
def handle_user_input():
    user_input = st.session_state.user_input
    if user_input:
        st.session_state.messages.append({"role": "user", "content": user_input})
        # 这里添加处理用户输入的逻辑
        response = "这是一个示例回复"  # 替换为实际的响应逻辑
        st.session_state.messages.append({"role": "assistant", "content": response})
        st.session_state.user_input = ""  # 清空输入框
