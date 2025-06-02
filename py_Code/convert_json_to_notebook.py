#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
将JSON文件转换为Jupyter Notebook格式

使用方法:
python convert_json_to_notebook.py
"""

import json
import shutil
import os

def convert_json_to_notebook(json_file_path, output_notebook_path):
    """
    将JSON文件转换为Jupyter Notebook格式
    
    参数:
    json_file_path: 输入的JSON文件路径
    output_notebook_path: 输出的notebook文件路径
    """
    try:
        # 读取JSON文件
        with open(json_file_path, 'r', encoding='utf-8') as f:
            notebook_data = json.load(f)
        
        # 验证是否是有效的notebook格式
        if 'cells' not in notebook_data:
            print("错误: JSON文件不包含notebook所需的'cells'字段")
            return False
        
        # 添加必要的notebook元数据（如果缺失）
        if 'metadata' not in notebook_data:
            notebook_data['metadata'] = {
                "kernelspec": {
                    "display_name": "Python 3",
                    "language": "python",
                    "name": "python3"
                },
                "language_info": {
                    "codemirror_mode": {
                        "name": "ipython",
                        "version": 3
                    },
                    "file_extension": ".py",
                    "mimetype": "text/x-python",
                    "name": "python",
                    "nbconvert_exporter": "python",
                    "pygments_lexer": "ipython3",
                    "version": "3.8.5"
                }
            }
        
        if 'nbformat' not in notebook_data:
            notebook_data['nbformat'] = 4
        
        if 'nbformat_minor' not in notebook_data:
            notebook_data['nbformat_minor'] = 4
        
        # 保存为.ipynb文件
        with open(output_notebook_path, 'w', encoding='utf-8') as f:
            json.dump(notebook_data, f, ensure_ascii=False, indent=2)
        
        print(f"成功转换！")
        print(f"输入文件: {json_file_path}")
        print(f"输出文件: {output_notebook_path}")
        return True
        
    except FileNotFoundError:
        print(f"错误: 找不到文件 {json_file_path}")
        return False
    except json.JSONDecodeError as e:
        print(f"错误: JSON格式无效 - {e}")
        return False
    except Exception as e:
        print(f"转换过程中出现错误: {e}")
        return False

def main():
    """主函数"""
    # 定义文件路径
    json_file = "Data/test.json"
    notebook_file = "LDA_Topic_Analysis_from_JSON.ipynb"
    
    print("开始转换JSON文件为Jupyter Notebook...")
    print("=" * 50)
    
    # 检查输入文件是否存在
    if not os.path.exists(json_file):
        print(f"错误: 输入文件 {json_file} 不存在")
        return
    
    # 执行转换
    success = convert_json_to_notebook(json_file, notebook_file)
    
    if success:
        print("=" * 50)
        print("转换完成！")
        print(f"\n现在你可以使用以下方式打开notebook:")
        print(f"1. 在Jupyter Lab中打开: {notebook_file}")
        print(f"2. 在VS Code中打开: {notebook_file}")
        print(f"3. 在命令行中运行: jupyter notebook {notebook_file}")
        
        # 显示文件信息
        file_size = os.path.getsize(notebook_file)
        print(f"\n文件信息:")
        print(f"- 文件大小: {file_size} 字节")
        print(f"- 文件位置: {os.path.abspath(notebook_file)}")
    else:
        print("转换失败！")

if __name__ == "__main__":
    main() 