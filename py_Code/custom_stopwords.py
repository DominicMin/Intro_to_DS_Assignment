# 自定义停用词列表（从LDA噪音主题提取）
custom_stopwords = [
    # 来自主题1: 粗俗/负面情绪表达
    'white', 'child', 'dog', 'date', 'brain', 
    'drink', 'bitch', 'dude', 'body', 'black',
    'sex', 'parent', 'instead', 'car', 'suck',
    'sense', 'food', 'single', 'half', 'stupid',
    
    # 来自主题8: 社交媒体互动
    'j', 'acc', 'll', 'gt', 'rlly',
    'lesbian', 'reply', 'slay', 'shut', 'yall',
    'goodnight', 'pfp', 'priv', 'twt', 'block',
    'ship', 'yea', 'bye', 'layout', 'pride',
    
    # 来自主题11: 粗俗表达
    'express', 'address', 'disbelief', 'informal', 'nah',
    'yall', 'bitch', 'mf', 'dude', 'lmfao',
    'gon', 'tf', 'shut', 'imma', 'mad',
    'll', 'yea', 'na', 'yo', 'nigga',
    
    # 来自主题16: 非英语内容
    'na', 'po', 'ko', 'ang', 'ako',
    'ka', 'lang', 'mo', 'ng', 'ba',
    'sa', 'naman', 'pa', 'huhu', 'si',
    'talaga', 'yung', 'yan', 'congrats', 'pogi'
]
