import re
import json

with open("py_Code\contractions.json",'r',encoding='utf-8') as f:
    contractions_map=json.load(f)


def text_expand(original_string, contraction_mapping=contractions_map):
    print(f"\n--- Processing original string: '{original_string}' ---") # 调试输出

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
        
        print(f"  Matched: '{old_text}' (lower: '{old_text.lower()}') -> Mapped to: '{new_text}'") # 调试输出
        
        if new_text:
            return new_text + " "
        else:
            print(f"  Warning: Matched '{old_text}' but not found in mapping. Returning original.")
            return old_text + " "
    
    expanded_string = contractions_pattern.sub(
        repl=text_mapping,
        string=original_string
    )
    final_result = expanded_string.strip()
    print(f"--- Final expanded string: '{final_result}' ---\n") # 调试输出
    return final_result



mock_data_df = {
    "posts": [
        ["I'm going to the party."], # 句首的 I'm
        ["Hey ur going to the party? U r so cool! Lol, btw I'm gonna be there too. Np."], # 句中的 I'm 和其他
        ["This is a furry animal. During the period, I found ur bag."],
        ["UR coming with Us, BTW? It's a fun day."],
        ["No contractions here."]
    ]
}

print(text_expand('omg omgt.'))