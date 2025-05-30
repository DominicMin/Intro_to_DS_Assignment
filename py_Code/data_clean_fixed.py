import re
import json
import pandas as pd

class Data_to_Clean_Fixed:
    # Load the contraction map in class
    with open("py_Code/contractions.json", mode='r', encoding='utf-8') as f:
        contractions_map = json.load(f)
    
    def __init__(self, source):
        self.data = source.copy()
    
    @staticmethod
    def text_expand(original_string, contraction_mapping=None):
        if contraction_mapping is None:
            contraction_mapping = Data_to_Clean_Fixed.contractions_map
            
        # 清理映射值中的多余空格
        standardized_contraction_map = {
            k.lower(): v.strip() for k, v in contraction_mapping.items()
        }

        sorted_contractions = sorted(
            standardized_contraction_map.items(),
            key=lambda item: len(item[0]),
            reverse=True
        )

        pattern_parts = []
        for contraction, _ in sorted_contractions:
            pattern_parts.append(r'\b' + re.escape(contraction) + r'\b')

        if not pattern_parts:
            return original_string

        contractions_pattern = re.compile(
            '({})'.format('|'.join(pattern_parts)),
            flags=re.IGNORECASE
        )

        def text_mapping(match_obj):
            old_text = match_obj.group(0)
            new_text = standardized_contraction_map.get(old_text.lower())
            
            if new_text:
                return new_text  # 不再添加额外空格
            else:
                return old_text
        
        expanded_string = contractions_pattern.sub(
            repl=text_mapping,
            string=original_string
        )
        
        # 清理多余的空格
        final_result = re.sub(r'\s+', ' ', expanded_string).strip()
        return final_result
    
    def expand_contractions(self):
        def process_expand_contractions(original_list):
            processed_list = []
            for sentence in original_list:
                expanded_sentence = Data_to_Clean_Fixed.text_expand(sentence)
                processed_list.append(expanded_sentence)
            return processed_list
        self.data["posts"] = self.data["posts"].apply(process_expand_contractions)


# 测试修正后的函数
def test_fixed_function():
    test_sentences = [
        "I'll go to the store",
        "I'm very happy today", 
        "They're coming tomorrow",
        "Don't worry about it",
        "Can't believe it's true",
        "won't be able to make it",
        "She's really nice",
        "We're going home",
        "It's a beautiful day",
        "You're amazing"
    ]

    print("修正后的连接词展开功能测试:")
    print("=" * 50)

    for sentence in test_sentences:
        expanded = Data_to_Clean_Fixed.text_expand(sentence)
        print(f"原文: {sentence}")
        print(f"展开: {expanded}")
        print("-" * 30)

if __name__ == "__main__":
    test_fixed_function() 