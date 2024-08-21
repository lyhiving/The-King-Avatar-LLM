"""
    目标：将大段文档通过gpt3.5识别变成一问一答的问答对。
    流程：1.gpt自动获取合适的问题；2.gpt自动根据问题和文档生成问答对。
    优点：几乎无需人工介入，自动获取问题，自动根据问题生成问答对。
    缺点：受限于大模型输入长度限制，可能无法一次性输入全部文档。
    建议：使用gpt3.5-16k可以一次输入大量文本，文档最好不超过5000字。

    FAQ：
    1.Q：gpt两个步骤是否可以合并成一个请求让gpt返回，可以节省约一半的时间和tokens？
      A：拆成两次主要是因为问题可能需要人工微调修改后再去生成答案，这样可以提高知识库质量，当然也可以全部自动处理。
    2.Q：大模型有字数限制无法大文档一次输入？
      A：目前这个没有好的解决办法，只能通过预先拆分大文档为多个文档片段后分批执行。
"""
import datetime
import time
import requests
import openai
from openai import OpenAI
from tqdm import tqdm
import re

q_content = "帮我生成适合作为问答对的问题。"
a_content = "帮我拼成问答对。"

prompt1 = '''
#01 你是一个问答对数据集处理专家。

#02 你的任务是根据我给出的内容，生成适合作为问答对数据集的问题。

#03 问题要尽量短，不要太长。

#04 一句话中只能有一个问题。

#05 生成的问题必须宏观、价值，不要生成特别细节的问题。问题不能包含章节标题，也不要包含“第几章”或者“第几节”这样的提示。

#06 生成问题示例：

"""

阿莫西林的特点有哪些方面？

介绍一下西地那非。

"""

#07 以下是我给出的内容：

"""

{{此处替换成你的内容}}

"""
'''

prompt2 = '''
#01 你是一个问答对数据集处理专家，主要的领域在医学、药学、护理等卫生健康领域。

#02 你的任务是根据我的问题和我给出的内容，生成对应的问答对。

#03 答案要全面，多使用我的信息，内容要更丰富。同时你不能提问类似“这一章主要讨论了什么内容？”这样的问题。也不能提问类似“这一节的重点”，因为科目比较多，无法通过章节的标题进行考核。

#04 你必须根据我的问答对示例格式来生成：

"""

{"question": "热原是什么？", "answer": "热原（pyrogen）是微生物产生的-种内毒素（endotoxin），它是能引起恒温动物体温异常升高的致热物质。大多数细菌都能产生热原，其中致热能力最强的是革兰阴性杆菌。霉菌甚至病毒也能产生热原。"}

{"question": "基金是什么？", "answer": "基金，广义是指为了某种目的而设立的具有一定数量的资金。主要包括公积金、信托投资基金、保险基金、退休基金，各种基金会的基金。从会计角度透析，基金是一个狭义的概念，意指具有特定目的和用途的资金。我们提到的基金主要是指证券投资基金。"}

#05 我的问题如下：

"""

{{此处替换成你上一步生成的问题}}

"""

#06 我的内容如下：

"""

{{此处替换成你的内容}}

"""
'''

def split_chapters(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        content = file.read()

    # 用正则表达式匹配章标题

    chapters = re.split(r'(第[一二三四五六七八九十百千万]*[章节])', content)

    # 合并章标题和内容
    chapter_list = []
    for i in range(1, len(chapters), 2):
        chapter_title = chapters[i]
        chapter_content = chapters[i + 1] if i + 1 < len(chapters) else ""
        chapter_list.append(chapter_content)
        # chapter_list.append(chapter_title + chapter_content)

    return chapter_list


# 示例用法


def generate_question(MODEL_NAME, prompt1,text):
    # 这里采用dashscope的api调用模型推理，通过http传输的json封装返回结果
    openai_api_key = "EMPTY"
    openai_api_base = "http://localhost:11434/v1"
    client = OpenAI(
        api_key=openai_api_key,
        base_url=openai_api_base,
    )
    
    # print(client.models.list())
    model_name = client.models.list().data[0].id
    # print(model_name)
    prompt = prompt1.replace("{{此处替换成你的内容}}", text)
    completion = client.chat.completions.create(
      model=model_name,
      messages=[
                {"role": "system", "content": prompt},
                {"role": "user", "content": q_content}
      ]
    )
    start_time = time.time()
    print(completion)
    print("耗时", time.time() - start_time)
    return completion.choices[0].message.content

def generate_qa(MODEL_NAME, prompt2, text , question_text=None):
    # 这里采用dashscope的api调用模型推理，通过http传输的json封装返回结果
    openai_api_key = "EMPTY"
    openai_api_base = "http://localhost:11434/v1"
    client = OpenAI(
        api_key=openai_api_key,
        base_url=openai_api_base,
    )
    prompt = prompt2.replace("{{此处替换成你上一步生成的问题}}", question_text).replace("{{此处替换成你的内容}}", text)
    completion = client.chat.completions.create(
      model=MODEL_NAME,
      messages=[
                {"role": "system", "content": prompt},
                {"role": "user", "content": a_content}
      ]
    )
    start_time = time.time()

    print("耗时:", time.time() - start_time)
    return completion.choices[0].message.content



def write_to_file(content):
    # timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
    file_name = f"gen_QA.txt"
    with open(file_name, "a+") as file:
        file.write(content)
    print("File 'new_file.txt' has been created and written.")


# def read_file(file_name):
#     try:
#         with open(file_name, "r") as file:
#             content = file.read()
#         return content
#     except FileNotFoundError:
#         print(f"File '{file_name}' not found.")


def main():
    # MODEL_NAME = "wangrongsheng/mistral-7b-v0.3-chinese"
    MODEL_NAME = "qwen2"

    file_path = './data/2024年药学专业知识（一）指南OCR.md'
    text_content = split_chapters(file_path)
    for text in tqdm(text_content):
        # print('MODEL_NAME:\n', MODEL_NAME)
        # print('prompt1:\n', prompt1)
        # print('text:\n', text)
        
        question_text = generate_question(MODEL_NAME, prompt1, text)
        print('text:\n', question_text)

        qa_text = generate_qa(MODEL_NAME, prompt2, text, question_text)
        print('qa_text\n', qa_text)

        write_to_file(qa_text)
    print("结束进程！")

main()

