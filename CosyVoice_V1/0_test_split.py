import re

def split_text(text, max_words=30):
    # 使用正则表达式根据标点符号切分文本
    # 这里匹配的标点符号包括句号、逗号、问号、叹号、分号、冒号
    sentences = re.split(r'([，,.。！？、；：])', text)
    # 将分割的句子重新组合
    sentences = [s.strip() + (sentences[i+1] if i+1 < len(sentences) else '') for i, s in enumerate(sentences) if i % 2 == 0]
    
    # 用来保存最终切分后的文本片段
    result = []
    current_chunk = []
    current_word_count = 0
    
    # breakpoint()

    # 遍历句子并按最大词数分段
    for sentence in sentences:
        word_count = len(sentence)
        
        # 如果当前句子加上当前片段的词数超过限制，就开始新的片段
        if current_word_count + word_count > max_words:
            result.append(" ".join(current_chunk))
            current_chunk = [sentence]
            current_word_count = word_count
        else:
            current_chunk.append(sentence)
            current_word_count += word_count
    
    # 添加最后一个片段
    if current_chunk:
        result.append(" ".join(current_chunk))
    
    return result

# 示例
text = "这是一个示例文本。这个文本包括多个句子，我们将其按照标点符号切分。确保每个片段包含少于50个词。你可以使用这个脚本来处理更长的文本。"

result = split_text(text)

# 输出结果
for idx, chunk in enumerate(result, 1):
    print(f"{chunk}\n")