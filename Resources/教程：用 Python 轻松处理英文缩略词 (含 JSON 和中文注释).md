## **教程：用 Python 轻松处理英文缩略词**

大家好！在处理英文文本的时候，我们经常会遇到像 "I'm", "don't", "you're" 这样的缩略词 (contractions)。它们是口语和非正式书写中很常见，但在做文本分析（比如统计词语、情感分析）时，把它们展开成原始形式（比如 "I am", "do not", "you are"）通常会让处理更简单、结果更准确。

这篇教程会教你如何用 Python，结合 JSON 文件和正则表达式，来自动地展开这些缩略词。

**目标：**

* 学习什么是缩略词以及为什么要处理它们。  
* 将缩略词和它们的完整形式存储在 JSON 文件中。  
* 用 Python 读取 JSON 文件。  
* 使用 Python 的 re 模块（正则表达式）来查找并替换文本中的缩略词。

**需要准备：**

* Python 3 环境。  
* 一点点 Python 基础知识（变量、字典、函数、文件操作）。

### **第一步：准备缩略词“字典” (JSON 文件)**

首先，我们需要一个列表，告诉程序哪个缩略词对应哪个完整形式。就像查字典一样。为了方便管理和复用，我们把这个“字典”存成一个 JSON 文件。

JSON (JavaScript Object Notation) 是一种轻量级的数据交换格式，它非常适合存储像 Python 字典这样的结构化数据。

创建一个名为 contractions\_map.json 的文件，内容如下：
```json

{  
    "ain't": "is not",  
    "aren't": "are not",  
    "can't": "cannot",  
    "can't've": "cannot have",  
    "'cause": "because",  
    "could've": "could have",  
    "couldn't": "could not",  
    "didn't": "did not",  
    "doesn't": "does not",  
    "don't": "do not",  
    "hadn't": "had not",  
    "hasn't": "has not",  
    "haven't": "have not",  
    "he'd": "he would",  
    "he'll": "he will",  
    "he's": "he is",  
    "how'd": "how did",  
    "how'll": "how will",  
    "how's": "how is",  
    "i'd": "i would",  
    "i'll": "i will",  
    "i'm": "i am",  
    "i've": "i have",  
    "isn't": "is not",  
    "it'd": "it would",  
    "it'll": "it will",  
    "it's": "it is",  
    "let's": "let us",  
    "ma'am": "madam",  
    "might've": "might have",  
    "must've": "must have",  
    "mustn't": "must not",  
    "needn't": "need not",  
    "shan't": "shall not",  
    "she'd": "she would",  
    "she'll": "she will",  
    "she's": "she is",  
    "should've": "should have",  
    "shouldn't": "should not",  
    "that'd": "that would",  
    "that's": "that is",  
    "there'd": "there would",  
    "there's": "there is",  
    "they'd": "they would",  
    "they'll": "they will",  
    "they're": "they are",  
    "they've": "they have",  
    "wasn't": "was not",  
    "we'd": "we would",  
    "we'll": "we will",  
    "we're": "we are",  
    "we've": "we have",  
    "weren't": "were not",  
    "what're": "what are",  
    "what's": "what is",  
    "what've": "what have",  
    "where's": "where is",  
    "who'll": "who will",  
    "who's": "who is",  
    "who've": "who have",  
    "why's": "why is",  
    "won't": "will not",  
    "would've": "would have",  
    "wouldn't": "would not",  
    "y'all": "you all",  
    "you'd": "you would",  
    "you'll": "you will",  
    "you're": "you are",  
    "you've": "you have"  
}

*小提示：* 这个列表尽可能全，但你也可以根据需要自己添加或修改。注意键（缩略词）最好用小写，这样后面匹配时更方便。
```

### **第二步：编写 Python 代码**

现在，我们来写 Python 代码，完成读取 JSON、查找和替换缩略词的任务。
```python

\# 导入需要的模块  
import json  \# 用于处理 JSON 文件  
import re    \# 用于处理正则表达式  
import os    \# 用于检查文件是否存在

# \--- (A) 读取 JSON 文件 \---  
def load\_contraction\_map(filename='contractions\_map.json'):  
    """  
    从 JSON 文件加载缩略词映射字典。

    Args:  
        filename (str): JSON 文件的路径。

    Returns:  
        dict: 包含缩略词到展开形式映射的字典。如果文件不存在或读取失败，返回空字典。  
    """  
    contraction\_map \= {} \# 初始化一个空字典  
    if os.path.exists(filename): \# 检查文件是否存在  
        try:  
            \# 使用 'r' (读取) 模式和 utf-8 编码打开文件  
            with open(filename, 'r', encoding='utf-8') as f:  
                \# 使用 json.load() 从文件读取内容并解析成 Python 字典  
                contraction\_map \= json.load(f)  
            print(f"成功从 '{filename}' 加载缩略词字典。")  
        except json.JSONDecodeError:  
            print(f"错误：文件 '{filename}' 不是有效的 JSON 格式。")  
        except Exception as e:  
            print(f"读取文件 '{filename}' 时发生错误: {e}")  
    else:  
        print(f"警告：缩略词文件 '{filename}' 未找到。将使用空字典。")  
    return contraction\_map

\# \--- (B) 编译正则表达式模式 \---  
def compile\_contractions\_pattern(contraction\_map):  
    """  
    根据缩略词字典的键编译正则表达式模式。

    Args:  
        contraction\_map (dict): 缩略词映射字典。

    Returns:  
        re.Pattern: 编译后的正则表达式对象。如果字典为空，返回 None。  
    """  
    if not contraction\_map: \# 如果字典是空的  
        print("缩略词字典为空，无法编译正则表达式。")  
        return None

    \# 1\. 获取字典里所有的键 (就是所有的缩略词)  
    keys \= contraction\_map.keys()

    \# 2\. 用 '|' (逻辑或) 把所有缩略词连接起来，形成一个长长的模式字符串  
    \#    例如: "ain't|aren't|can't|..."  
    pattern\_string \= '|'.join(keys)

    \# 3\. 把上面生成的字符串放到括号里 '()'，构成一个捕获组  
    \#    例如: "(ain't|aren't|can't|...)"  
    \#    这表示我们要匹配括号里的任何一个词  
    full\_pattern\_string \= f"({pattern\_string})"

    \# 4\. 使用 re.compile() 编译这个模式字符串，得到一个正则表达式对象  
    \#    这样做可以提高后面重复匹配的效率  
    \#    flags=re.IGNORECASE 表示匹配时不区分大小写 (比如 "Don't" 和 "don't" 都能匹配)  
    \#    flags=re.DOTALL 让 '.' 可以匹配包括换行符在内的所有字符 (对缩略词影响不大，但通常加上也无妨)  
    contractions\_pattern \= re.compile(full\_pattern\_string, flags=re.IGNORECASE | re.DOTALL)

    print("正则表达式模式已成功编译。")  
    return contractions\_pattern

\# \--- (C) 定义替换函数 \---  
def expand\_match(contraction\_match, contraction\_map):  
    """  
    这个函数会被 re.sub() 在每次找到匹配时调用。  
    它接收匹配对象，并返回应该替换成的字符串。

    Args:  
        contraction\_match (re.Match): 正则表达式找到的匹配对象。  
        contraction\_map (dict): 缩略词映射字典。

    Returns:  
        str: 缩略词对应的展开形式，或者如果找不到则返回原始匹配文本。  
    """  
    \# 1\. 从匹配对象中获取实际匹配到的文本  
    \#    比如，如果匹配到 "I'm"，这里的 match\_text 就是 "I'm"  
    match\_text \= contraction\_match.group(0)

    \# 2\. 尝试用匹配文本的小写形式去字典里查找  
    \#    因为我们的字典键是小写的，并且正则设置了忽略大小写，所以用小写查找最稳妥  
    expanded \= contraction\_map.get(match\_text.lower())

    \# 3\. 如果找到了对应的展开形式  
    if expanded:  
        \# 就返回这个展开形式 (比如 "i am")  
        return expanded  
    else:  
        \# 4\. 如果小写形式没找到 (理论上不太可能发生，如果正则和字典是匹配的)  
        \#    作为备用方案，尝试用原始匹配文本直接查找 (处理字典里可能有大写键的特殊情况)  
        expanded \= contraction\_map.get(match\_text)  
        if expanded:  
            return expanded  
        else:  
            \# 5\. 如果实在找不到 (说明可能字典或正则有问题)，就返回原始匹配到的文本，避免报错  
            print(f"警告：在字典中找不到 '{match\_text}' 的展开形式。")  
            return match\_text

\# \--- (D) 定义主处理函数 \---  
def expand_contractions(text, contractions_pattern, contraction_map):  
    """  
    展开给定文本中的所有缩略词。

    Args:  
        text (str): 需要处理的原始文本。  
        contractions\_pattern (re.Pattern): 编译好的正则表达式对象。  
        contraction\_map (dict): 缩略词映射字典。

    Returns:  
        str: 展开缩略词后的文本。如果模式或字典无效，返回原始文本。  
    """  
    if not contractions\_pattern or not contraction\_map:  
        print("错误：正则表达式模式或缩略词字典无效，无法展开缩略词。")  
        return text # 返回原始文本

    # 使用 re.sub() 进行替换  
    # 参数1: 要查找的模式 (我们编译好的正则对象)  
    # 参数2: 一个函数，用来决定每次匹配到后替换成什么。  
    #        这里我们不能直接传 expand_match，因为它还需要 contraction_map 这个参数。  
    #        所以我们用 lambda 创建一个临时的简单函数，它接收匹配对象 m，  
    #        然后调用 expand_match(m, contraction_map)。  
    # 参数3: 需要处理的原始文本  
    expanded_text = contractions_pattern.sub(lambda m: expand_match(m, contraction_map), text)

    return expanded_text

\# \--- (E) 运行示例 \---  
if \_\_name\_\_ \== "\_\_main\_\_":  
    \# 1\. 加载缩略词字典  
    contraction\_dictionary \= load\_contraction\_map('contractions\_map.json')

    \# 2\. 编译正则表达式模式  
    pattern \= compile\_contractions\_pattern(contraction\_dictionary)

    \# 3\. 准备示例文本  
    sample\_text \= "I'm learning NLP, but it's challenging. We're using Python, and it doesn't handle contractions automatically. You're gonna see how we'll fix it. He's happy 'cause he'd finished."

    print("\\n--- 处理开始 \---")  
    print("原始文本:")  
    print(sample\_text)

    \# 4\. 调用函数展开缩略词 (只有在字典和模式都有效时才执行)  
    if pattern and contraction\_dictionary:  
        processed\_text \= expand\_contractions(sample\_text, pattern, contraction\_dictionary)  
        print("\\n处理后的文本:")  
        print(processed\_text)  
    else:  
        print("\\n无法进行缩略词展开，请检查之前的错误信息。")

    print("--- 处理结束 \---")

    \# 另一个例子，包含大小写和字典里没有的词  
    test\_text\_2 \= "She's nice, BUT HE'S not. They'll arrive soon. What's this? It isn't mine. Y'all ready?"  
    print("\\n--- 另一个例子 \---")  
    print("原始文本:")  
    print(test\_text\_2)  
    if pattern and contraction\_dictionary:  
        processed\_text\_2 \= expand\_contractions(test\_text\_2, pattern, contraction\_dictionary)  
        print("\\n处理后的文本:")  
        print(processed\_text\_2)
```

### **如何运行代码**

1. **保存 JSON 文件：** 将上面提供的 JSON 内容保存为 contractions\_map.json 文件。  
2. **保存 Python 代码：** 将上面提供的 Python 代码保存为 .py 文件（例如 contraction\_expander.py）。  
3. **放在同一目录：** 确保 .json 文件和 .py 文件在同一个文件夹（目录）下。  
4. **运行 Python 文件：** 在你的终端或命令行中，切换到该目录，然后运行 python contraction\_expander.py。

你将看到程序首先加载 JSON 文件，然后编译正则表达式，最后输出原始文本和处理后展开了缩略词的文本。

### **总结**

恭喜！你现在学会了如何：

* 用 JSON 文件来管理缩略词列表。  
* 用 Python 读取这个 JSON 文件。  
* 构建一个灵活的正则表达式来查找所有这些缩略词（忽略大小写）。  
* 编写一个替换函数来查找每个匹配到的缩略词的完整形式。  
* 使用 re.sub() 将文本中的所有缩略词替换掉。

这个方法可以很好地处理大多数常见的英文缩略词，是你进行文本预处理时非常有用的一个技巧！