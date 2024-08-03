import streamlit as st
from ragchat import Model_center
from llm import InternLM
from tts import text_to_speech
import os


def on_btn_click():
    del st.session_state.messages

#通过 @st.cache_resource 装饰器，load_model 函数只会在首次调用时执行，确保模型只加载一次。随后每次用户交互时，代码将使用已加载的模型。

@st.cache_resource
def load_model(model_name_or_path):
    llm = InternLM(model_path=model_name_or_path)
    model_center = Model_center(llm)
    return model_center

model_name_or_path = './quanzhigaoshou'

if not os.path.exists(base_path):

    os.system('apt install git')
    os.system('apt install git-lfs')
    os.system(f'git clone https://code.openxlab.org.cn/shiqiyioo/quanzhigaoshou.git {model_name_or_path}')
    os.system(f'cd {model_name_or_path} && git lfs pull')
    # print("模型下载完成")


# 加载模型，只执行一次
# model_name_or_path ='./model/InterLM_chat_7b'
model_center = load_model(model_name_or_path)

# model_center = Model_center(llm)
if "messages" not in st.session_state:
    st.session_state["messages"] = []     
# 在侧边栏中创建一个标题和一个链接
with st.sidebar:
    st.markdown("The-King-Avatar-LLM")
    "[InternLM](https://github.com/InternLM/InternLM.git)"
    "[chat-huyu-ABao](https://github.com/shiqiyio/The-King-Avatar-LLM)"
    "感谢[chat-huyu-ABao](https://github.com/hoo01/chat-huyu-ABao.git)"
    st.button('Clear Chat History', on_click=on_btn_click)

# 创建一个标题和一个副标题
st.title("The-King-Avatar-LLM")
st.caption("🚀 A streamlit chatbot powered by InternLM2_7B QLora")
    
# 遍历session_state中的所有消息，并显示在聊天界面上
for msg in st.session_state.messages:
    st.chat_message("user").write(msg["user"])
    st.chat_message("assistant").write(msg["assistant"])

# Get user input
if prompt := st.chat_input("提出任何关于《全职高手》的问题"):
    # Display user input
    st.chat_message("user").write(prompt)
        # 使用 qa_chain 生成回答
    response = model_center.qa_chain_self_answer(prompt)
    
    # 将问答结果添加到 session_state 的消息历史中
    st.session_state.messages.append({"user": prompt, "assistant": response}) 
    # 显示回答
    st.chat_message("assistant").write(response)

    if st.button("语音播放"):
        # st.write("点击了转换为语音按钮")  # Debugging line
        try:
            # Your TTS function should convert `response` text to speech
            print("准备转换为音频...")
            text_to_speech(response)
            # st.write("语音文件生成成功")  # Debugging line
            
            audio_file = './data/result.mp3'
            if os.path.exists(audio_file):
                st.audio(audio_file, format='audio/mp3')
            else:
                st.error("音频文件未生成或路径错误")
        except Exception as e:
            st.error(f"处理语音时发生错误: {e}")  # Error handling and debugging