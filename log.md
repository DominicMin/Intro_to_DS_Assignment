### 主题建模优化参数

- 词典过滤0.2，50
  发现一个特性：语言过滤模型在保留规范的英文标点使用时准确率很高
  于是优化了语言过滤策略

  - 语言过滤0.85

    - ![1748488968573](image/log/1748488968573.png)
    - ![1748489002083](image/log/1748489002083.png)
  - 优化过滤策略+语言过滤0.75
- 使用N-gram 未移除部分

  - ![1748531921649](image/log/1748531921649.png)
- 使用N-gram 移除部分
