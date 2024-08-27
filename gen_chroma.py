from langchain.text_splitter import TextSplitter
from langchain.schema import Document
from langchain_huggingface import HuggingFaceEmbeddings 
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import TextLoader
from pathlib import Path
import os
import sys
import re 
#指定两个文本文件的路径
file_path_1='./data/novel.txt'
folder_path = './data/input'
# file_path_2='./data/baozong_taici.txt'

embedding_function = HuggingFaceEmbeddings(model_name="shibing624/text2vec-base-chinese")

persist_directory ='./chroma' #路径如果报错就修改成绝对路径


class CustomMarkdownTextSplitter(TextSplitter):
    def __init__(self):
        super().__init__()

    def split_text(self, text: str):
        # 移除图片标记的正则表达式
        text = re.sub(r'!\[.*?\]\(.*?\)', '', text)

        # 匹配章节和小节标题的正则表达式
        chapter_pattern = re.compile(r'(第[一二三四五六七八九十百千万]+章[^#\n]*)', re.MULTILINE)
        section_pattern = re.compile(r'(第[一二三四五六七八九十百千万]+节[^#\n]*)', re.MULTILINE)

        # 存放分割后的文档部分
        splits = []
        last_end = 0

        # 找到所有章节标题的位置
        for match in chapter_pattern.finditer(text):
            chapter_title = match.group(1).strip()
            chapter_start = match.start()

            # 处理上一章节到这一章节之间的内容
            if last_end < chapter_start:
                content_before = text[last_end:chapter_start].strip()
                if content_before:
                    splits.append(content_before)
            
            # 添加当前章节标题
            splits.append(chapter_title)
            last_end = match.end()

            # 分割章节内的小节内容
            section_splits, last_section_end = self._split_sections(text[last_end:], section_pattern)
            splits.extend(section_splits)
            last_end += last_section_end

        # 处理最后一部分内容
        # if last_end < len(text):
        #     content_after = text[last_end:].strip()
        #     if content_after:
        #         splits.append(content_after)

        return splits

    def _split_sections(self, text: str, section_pattern: re.Pattern):
        """分割章节内部的小节"""
        splits = []
        last_end = 0

        for match in section_pattern.finditer(text):
            section_title = match.group(1).strip()
            section_start = match.start()

            if last_end < section_start:
                content_before = text[last_end:section_start].strip()
                if content_before:
                    splits.append(content_before)

            # 添加小节标题
            splits.append(section_title)
            last_end = match.end()

        # # 处理最后一部分内容
        # if last_end < len(text):
        #     content_after = text[last_end:].strip()
        #     if content_after:
        #         splits.append(content_after)

        return splits, last_end

def process_markdown_file(file_path,my_meta):
    # 加载文本文件
    loader = TextLoader(file_path, encoding='utf-8')
    docs = loader.load()

    # 初始化自定义的Markdown文本分割器
    splitter = CustomMarkdownTextSplitter()

    # 对每个文档进行分割
    split_docs = []
    for doc in docs:
        split_chunks = splitter.split_text(doc.page_content)
        for i in range(len(split_chunks)):
            _doc = Document(page_content=split_chunks[i], metadata=my_meta)
            split_docs.append(_doc)

    return split_docs
    

def generate_split_docs(folder_path):
    directory_path = Path(folder_path)
    # 遍历目录中的文件
    for file_path in directory_path.rglob('*'):  # 使用 rglob('*') 递归遍历所有文件
        if file_path.is_file():
            print(f"[初始化]: {file_path}" )
            my_meta = {
                'case': file_path.parts[-2],
                'filename': file_path.stem,
                'suffix': file_path.suffix.lower(),
                'source':file_path.name
            }
            split_docs = process_markdown_file(file_path,my_meta)
            # print(split_docs)
            vectordb = Chroma.from_documents(
                documents=split_docs,
                embedding=embedding_function,
                persist_directory=persist_directory,
                collection_name='books'
            )
            print(f"[索引中] {file_path} NUM: {len(split_docs)}")
            vectordb.persist()
            print(f"[持久化] {file_path} NUM: {len(split_docs)}")
    return split_docs

# 当直接运行脚本时执行的代码
if __name__ == "__main__":
    split_docs = generate_split_docs(folder_path)
    print("文本分割和向量数据库构建完成，并已持久化到磁盘。")

# 导出split_docs变量
__all__ = ['generate_split_docs']