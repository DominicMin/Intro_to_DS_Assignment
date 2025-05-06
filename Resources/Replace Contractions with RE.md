---
mindmap-plugin: basic
source: "[[教程：用 Python 轻松处理英文缩略词 (含 JSON 和中文注释)]]"
---

# Contraction Replacement

## compile re pattern
- 需要读取字典中的值
- 编译好之后是一个regex object，可以直接调用sub方法进行替换
    - 现在需要处理sub方法中的repl参数
    - 这需要一个新的函数，将原始text映射到应该替换的repl
        - 设计expand_match( )函数
            - Args
                - 正则表达式找到的匹配对象（re.Match）
                    - 这个对象怎么传递：lambda小函数
                        - 好的，这个问题问得很好！`lambda` 函数本身并不“主动”去接收 `re.Match` 对象，而是 `re.sub()` 函数在内部机制中**主动将** `re.Match` 对象**传递给** `lambda` 函数。
                           具体过程是这样的：
                           1. **`re.sub()` 查找匹配：** 当你调用 `contractions_pattern.sub(..., text)` 时，`re.sub` 会根据 `contractions_pattern` 在 `text` 字符串里查找第一个匹配项（比如找到了 "I'm"）。
                           2. **创建 `re.Match` 对象：** 一旦找到匹配项，`re.sub` 会在内部创建一个 `re.Match` 对象。这个对象包含了这次匹配的所有信息（匹配到的文本 "I'm"、在原字符串中的位置等）。
                           3. **检查替换参数类型：** `re.sub` 检查它的第二个参数。它发现这个参数不是一个简单的字符串，而是一个函数（也就是我们写的 `lambda m: expand_match(m, contraction_map)`）。
                           4. **调用替换函数并传递参数：** 因为替换参数是个函数，`re.sub` 的设计就是：**它会调用这个函数，并且把第 2 步中创建的那个 `re.Match` 对象作为第一个（也是唯一的）参数传递给这个函数。**
                           5. **`lambda` 接收参数：** 我们定义的 `lambda m: ...` 声明了一个接收单个参数的匿名函数，参数名叫 `m`。当 `re.sub` 调用这个 `lambda` 函数并把 `re.Match` 对象传进来时，这个对象就被赋值给了参数 `m`。
                           6. **执行 `lambda` 体：** `lambda` 函数内部的代码 `expand_match(m, contraction_map)` 被执行。这时，`m` 就持有那个 `re.Match` 对象，可以传递给 `expand_match` 函数使用了。
                           **简单地说：** 不是 `lambda` 去“拿” `re.Match` 对象，而是 `re.sub` 在找到匹配后，按照约定“塞给” `lambda` 函数一个 `re.Match` 对象作为输入。`lambda m: ...` 只是定义了一个准备好接收这个输入的“入口”（参数 `m`）。
                - 之前的字典，用来实现text-repl的映射
            - Returns
                - 用来替换的str，即我们需要的repl
            - 流程
                - 使用group（0）从对象中调取原始text
                    - 好的，我们来解释一下 `match_text = contraction_match.group(0)` 这行代码：
                       在 `expand_match` 函数中，参数 `contraction_match` 是一个由 `re.sub` 传递过来的**正则表达式匹配对象 (Match Object)**。这个对象包含了关于当前找到的那个缩略词的所有信息。
                       `.group(0)` 是这个匹配对象的一个方法（函数），它的作用是：
                       **返回整个正则表达式模式实际匹配到的那部分文本字符串。**
                       **举个例子：**
                       假设你的正则表达式模式是 `(ain't|i'm|don't)`，并且它在处理文本 "Hi, I'm Bob." 时，成功匹配到了 "I'm"。
                       1. `re.sub` 找到 "I'm" 这个匹配。
                       2. 它调用 `expand_match` 函数，并将代表这次匹配的 `contraction_match` 对象传给它。
                       3. 在 `expand_match` 函数内部，执行 `contraction_match.group(0)`。
                       4. 这个 `.group(0)` 调用就会返回字符串 `"I'm"`。
                       5. 然后这行代码 `match_text = contraction_match.group(0)` 就把字符串 `"I'm"` 赋值给了 `match_text` 这个变量。
                       **简单来说：** `contraction_match.group(0)` 就是用来**提取出当前具体匹配到的那个缩略词文本**，以便后续在 `contraction_map` 字典里查找它的展开形式。
                       * `.group()` 方法还可以有其他用法，比如 `group(1)`, `group(2)` 等，用来获取正则表达式中用括号 `()` 定义的**捕获组**匹配到的内容。但 `.group(0)` 总是代表整个匹配到的字符串。
                - 使用字典get（）进行映射
                    - 初始使用小写形式替换
                    - 分类讨论，避免报错
                        - if 映射成功：直接返回repl
                        - else 使用原始形式
                            - if 映射成功：返回
                            - else 直接返回原始text