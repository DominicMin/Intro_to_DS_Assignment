
在数据分析中，我们经常需要按某些标准将数据分组（groupby），然后对这些组进行操作。有时，我们希望得到每个组的汇总信息（比如求和、平均值），这可以看作是一种“合并”组内数据的方式。另一些时候，我们可能想把基于组的计算结果添加回原始数据中。

本教程将介绍 Pandas 中几种常见的处理和“合并”分组数据的方法。

核心概念：

- groupby(): Pandas 中用于数据分组的核心函数。
    
- 聚合 (Aggregation): 将一个组的数据缩减为一个或多个汇总统计量（如求和、平均值、计数）。常用的函数有 agg(), sum(), mean(), count() 等。
    
- 转换 (Transformation): 对一个组应用一个函数，返回与该组具有相同形状（相同索引）的结果。常用函数是 transform()。
    
- 应用 (Apply): 对一个组应用一个更通用的函数，可以返回标量、Series 或 DataFrame。常用函数是 apply()。
    
- 合并 (Merge/Join): 将不同的 DataFrame（可能包含分组操作的结果）基于共同的键连接起来。常用函数是 pd.merge() 或 DataFrame 的 join() 方法。
    

### 场景一：聚合组——将组内数据汇总成新表

这是最常见的“合并组”的理解：计算每个组的描述性统计数据，从而得到一个更小、更浓缩的 DataFrame，其中每一行代表一个原始组。

方法：使用 .agg() 或特定的聚合函数
```python
import pandas as pd  
import numpy as np  

# 创建一个示例 DataFrame  
data = {  
    '部门': ['销售部', '技术部', '销售部', '技术部', '市场部', '销售部', '技术部', '市场部'],  
    '员工': ['张三', '李四', '王五', '赵六', '孙七', '周八', '吴九', '郑十'],  
    '工资': [5000, 8000, 5500, 9000, 6000, 4800, 8500, 6200],  
    '项目数量': [3, 5, 2, 6, 3, 2, 5, 4]  
}  
df = pd.DataFrame(data)  
  
print("原始 DataFrame:")  
print(df)  
print("-" * 30)  
  
# 按“部门”分组  
grouped_by_dept = df.groupby('部门')  
  
# 1. 计算每个部门的平均工资和总项目数量  
# 使用 .agg() 并传入一个字典，指定对哪些列应用哪些函数  
dept_summary = grouped_by_dept.agg(  
    平均工资=('工资', 'mean'),      # 对“工资”列求平均值  
    总项目数量=('项目数量', 'sum'),  # 对“项目数量”列求和  
    员工人数=('员工', 'count')      # 对“员工”列计数  
)  
print("部门汇总信息 (使用 agg):")  
print(dept_summary)  
print("-" * 30)  
  groupe
# 2. 对不同列应用不同聚合，也可以对同一列应用多个聚合  
dept_multi_agg = grouped_by_dept['工资'].agg(['mean', 'sum', 'min', 'max'])  
print("各部门工资的多种统计:")  
print(dept_multi_agg)  
print("-" * 30)  
  
# 3. 直接使用聚合函数 (更简洁，但灵活性稍低)  
avg_salary_per_dept = grouped_by_dept['工资'].mean()  
print("各部门平均工资 (直接使用 .mean()):")  
print(avg_salary_per_dept)  
  
```
说明：

- df.groupby('部门') 创建了一个 DataFrameGroupBy 对象。
    
- .agg() 方法非常灵活，允许你为不同的列指定不同的聚合函数。结果是一个新的 DataFrame，其索引是分组的键（这里是“部门”）。
    
- 你也可以直接在分组后的 Series 或 DataFrame 上调用如 .sum(), .mean(), .count() 等函数。
    

### 场景二：将分组计算结果合并回原始 DataFrame

有时，我们不想得到一个汇总表，而是想把基于组的计算结果（比如组内平均值、组内排名）作为新的一列添加回原始的 DataFrame 中。

方法一：使用 .transform()

transform() 方法会将一个函数应用于每个组，然后返回一个与原始 DataFrame 具有相同索引的 Series 或 DataFrame。这使得它非常适合将结果直接赋给新列。

```python
# (接上一个代码块的 df)  
  
# 1. 计算每个员工所在部门的平均工资，并添加到原始 DataFrame  
df['部门平均工资'] = grouped_by_dept['工资'].transform('mean')  
print("添加了部门平均工资的 DataFrame (使用 transform):")  
print(df)  
print("-" * 30)  
  
# 2. 计算每个员工工资与部门平均工资的差额  
df['工资与部门平均差'] = df['工资'] - df['部门平均工资']  
print("添加了工资差额的 DataFrame:")  
print(df)  
print("-" * 30)  
  
# 3. 组内标准化 (例如：(值 - 均值) / 标准差)  
df['工资标准化(组内)'] = grouped_by_dept['工资'].transform(lambda x: (x - x.mean()) / x.std())  
print("工资组内标准化:")  
print(df)  
print("-" * 30)  
  
```

说明：

- grouped_by_dept['工资'].transform('mean') 对于“销售部”的每一行，都会返回销售部的平均工资；对于“技术部”的每一行，都会返回技术部的平均工资。
    
- 结果的索引与 df 的索引一致，所以可以直接赋值创建新列。    

方法二：先聚合，再使用 pd.merge()

如果你先创建了一个汇总表（如场景一中的 dept_summary），然后想把这个汇总表的信息合并回原始 DataFrame，可以使用 pd.merge()。

```python
# (接上一个代码块的 df 和 dept_summary)  
  
# dept_summary 的索引是“部门”，我们需要将其重置为列才能进行 merge  
dept_summary_for_merge = dept_summary.reset_index()  
# print("\n用于合并的部门汇总信息:")  
# print(dept_summary_for_merge)  
  
# 将部门汇总信息合并回原始 DataFrame  
# left_on 和 right_on 指定了连接的键  
df_merged_summary = pd.merge(  
    df,  
    dept_summary_for_merge[['部门', '平均工资', '总项目数量']], # 选择需要的列进行合并  
    left_on='部门',      # 左边 DataFrame (df) 的连接键  
    right_on='部门',     # 右边 DataFrame (dept_summary_for_merge) 的连接键  
    suffixes=('_员工', '_部门汇总') # 如果有重名列，添加后缀区分  
)  
# 注意：上面 dept_summary 中的“平均工资”列名与 df 中通过 transform 创建的“部门平均工资”不同。  
# 如果 merge 的列名与 df 中已有的列名相同，suffixes 参数会起作用。  
# 这里我们特意重命名了 dept_summary 中的列，或者只选取不冲突的列。  
  
# 为了演示suffixes，我们假设 dept_summary_for_merge 中有一个 '工资' 列  
# temp_summary = pd.DataFrame({'部门': ['销售部', '技术部'], '工资': [5100, 8500]})  
# df_merged_with_suffix = pd.merge(df, temp_summary, on='部门', suffixes=('_员工原始', '_部门平均'))  
# print(df_merged_with_suffix)  
  
  
print("合并了部门汇总信息的 DataFrame (使用 merge):")  
print(df_merged_summary.head()) # 显示前几行  
print("-" * 30)  
  
```

说明：

- pd.merge() 是一个功能强大的函数，用于基于一个或多个键将两个 DataFrame 的行连接起来，类似于 SQL 中的 JOIN 操作。
    
- 你需要确保用于连接的键在两个 DataFrame 中都存在。
    
- suffixes 参数用于处理合并后可能出现的同名列。

### 场景三：使用 .apply() 进行更复杂的分组操作

apply() 是 groupby 对象中最通用的方法。它可以接受一个函数，该函数在每个组上被调用一次，并且可以返回标量、Series 或 DataFrame。

如果 apply() 返回的是一个 Series 且其索引与原始分组的索引相同，或者返回一个与原始组具有相同索引的 DataFrame，其行为可能类似于 transform 或 agg 的组合。

```python
# (接上一个代码块的 df)  
  
# 示例：获取每个部门工资最高的员工信息  
def get_top_earner(group_df):  
    # group_df 是每个部门的子 DataFrame  
    return group_df.loc[group_df['工资'].idxmax()] # idxmax() 返回最大值的索引  
  
top_earners_per_dept = grouped_by_dept.apply(get_top_earner)  
print("各部门工资最高的员工信息 (使用 apply):")  
print(top_earners_per_dept)  
print("-" * 30)  
  
# 示例：对每个部门的工资进行排序，并取前N名 (这里取前2名)  
def get_top_n_salaries(group_df, n=2):  
    return group_df.nlargest(n, '工资')  
  
top_2_salaries_per_dept = grouped_by_dept.apply(get_top_n_salaries, n=2) # 可以传递额外参数给 apply 的函数  
# 结果会有一个额外的层级索引，是原始分组的索引  
print("各部门工资前2名的员工 (使用 apply):")  
print(top_2_salaries_per_dept)  
# 如果想去掉外层索引，可以 reset_index  
# print("\n去掉外层索引后:")  
# print(top_2_salaries_per_dept.reset_index(drop=True))  

```

说明：

- 传递给 apply() 的函数会接收每个组的 DataFrame 作为参数。
    
- apply() 的结果会根据你的函数返回什么而被智能地组合起来。

### 总结

Pandas 提供了多种方式来处理分组后的数据，并将结果以不同形式“合并”或展现：

- 若要得到每个组的汇总统计表，使用 groupby().agg() 或直接调用聚合函数如 groupby().sum()。
    
- 若要将基于组的计算结果（与原表等长）添加回原始 DataFrame 作为新列，groupby().transform() 是首选。
    
- 若已有一个汇总表，想将其信息根据共同键附加到原始表，使用 pd.merge()。
    
- 对于更复杂、自定义的分组逻辑，groupby().apply() 提供了最大的灵活性。
    

选择哪种方法取决于你的具体需求：是需要一个浓缩的摘要，还是需要在原始数据上扩充信息，或者是进行更复杂的组级运算。