import re
import json

# 加载连接词映射
with open("py_Code/contractions.json", "r", encoding='utf-8') as f:
    contractions_map = json.load(f)

def text_expand(original_string, contraction_mapping=contractions_map):
    standardized_contraction_map = {k.lower(): v for k, v in contraction_mapping.items()}

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
            return new_text + " "
        else:
            return old_text + " "
    
    expanded_string = contractions_pattern.sub(
        repl=text_mapping,
        string=original_string
    )
    final_result = expanded_string.strip()
    return final_result

# 测试用例
test_sentences = [
    "I'll go to the store",
    "i'll go",
    "omg",
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

print("测试连接词展开功能:")
print("=" * 50)

for sentence in test_sentences:
    expanded = text_expand(sentence)
    print(f"原文: {sentence}")
    print(f"展开: {expanded}")
    print("-" * 30)

# 检查JSON文件中的一些关键连接词
print("\n检查JSON文件中的关键连接词:")
print("=" * 50)

key_contractions = ["i'll", "i'm", "they're", "don't", "can't", "won't", "she's", "we're", "it's", "you're"]
for key in key_contractions:
    if key in contractions_map:
        print(f"{key} -> {contractions_map[key]}")
    else:
        print(f"{key} 不存在于映射中") 