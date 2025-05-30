---
mindmap-plugin: basic
---
# 仓库结构

## Assignment_Guidelines

- 储存作业要求

## Latex_Essay

- 储存论文工程文件

## py_Code

- Data

  - MBTI-QA
    - MBTI测试结果数据
  - mbti_1.csv
    - 某心理论坛帖子文本数据（弃用）
  - twitter_mbti.csv
    - Twitter posts with MBTI label （最终用于主题建模的数据）
  - /cleaned_data
    - 以pickle文件储存清洗后的数据
  - nltk_data
    - 储存数据清洗中nltk库需要用到的包
- /output

  - 储存可视化程序输出图片，主题建模结果等等
- contractions.json

  - 一个英文连接词&网络缩写字典，用于数据清洗
- data_clean.ipynb

  - 数据清洗程序
- data_clean.py

  - 储存数据清洗程序中的类，方便后续导入data_viewer.ipynb
- data_viewer.ipynb

  - 查看清洗后数据，并进行主题建模的程序
- visualization.py

  - MBTI测试结果可视化程序

## Resources

- 储存项目参考的资料，包括论文，AI回复等等

## 研究问题
1. MBTI人格测试结果在统计学上是否经得起考验
2. MBTI的4个维度是否相互独立
3. 不同MBTI人格在社交网站上的兴趣和行为是否有显著差异