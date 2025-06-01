import json
# 自定义停用词列表（从LDA噪音主题提取）
custom_stopwords = [
    # 来自主题1: 粗俗/负面情绪表达
    'white', 'drink', 'bitch', 'dude', 'body', 'black',
    'instead', 'suck', 'stupid',
    
    # 来自主题8: 社交媒体互动
    'j', 'acc', 'll', 'gt', 'rlly',
    'reply', 'slay', 'shut', 'yall',
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
    'talaga', 'yung', 'yan', 'congrats', 'pogi',
    
    # 从前300高频词中新增的停用词
    
    # 社交媒体平台和功能词汇
    'follow', 'followed', 'check', 'checked', 'stream',
    'automatically', 'post', 'posted', 'send', 'thread', 'tweets',
    'pic', 'photo', 'picture', 'pics', 'unfollowed', 'stalker',
    'rt', 'via', 'dm', 'app',
    
    # 感叹词和填充词
    'ah', 'hehe', 'huh', 'yay', 'yo', 'ya', 'nah', 'lmfao',
    'omg', 'wow', 'ugh', 'meh', 'hmm', 'oof', 'bruh',
    
    # 过于通用的形容词和副词
    'super', 'perfect', 'awesome', 'sweet', 'lovely', 'gorgeous',
    'extremely', 'especially', 'totally', 'seriously', 'truly',
    'completely', 'usually', 'often', 'basically', 'apparently',
    'genuinely', 'literally', 'actually', 'quite', 'rather',
    
    # 颜色词汇
    'red', 'blue', 'pink', 'white', 'black',
    
    # 时间相关通用词
    'yesterday', 'tonight', 'late', 'later',
    'early', 'minutes', 'hour', 'weeks', 'months',
    
    # 社交媒体行为和状态
    'tired', 'sick', 'dead', 'insane', 'mad', 'scared', 'confused',
    'missed', 'waiting', 'listening', 'eating', 'sleeping', 'woke',
    'forget', 'remember', 'tried', 'finished', 'started', 'welcome',
    
    # 家庭和人际关系通用词
    # 'dad', 'mom', 'sister', 'brother', 'parents', 'kids', 'kid',
    # 'girls', 'boys', 'woman', 'women', 'child', 'children',
    
    # 娱乐和媒体相关
    # 'album', 'songs', 'episode', 'season', 'concert', 'book', 'books',
    # 'story', 'dream', 'art', 'anime', 'series', 'shows', 'videos',
    # 'game', 'games', 'dance', 'music',
    
    # 单字母和简写
    'x', 'p', 'f', 'e', 'g', 'j', 'l', 'v', 're', 'st', 'nd', 'q'
    
    # 网络用语和缩写
    'lol', 'omg', 'tbh', 'imo', 'btw', 'fyi', 'aka', 'etc',
    'vs', 'ok', 'okay', 'yeah', 'yep', 'nope',
    
    # 过于通用的动词
    'cause', 'seems', 'goes', 'comes', 'giving', 'asked', 'says',
    'wants', 'needs', 'knows', 'meant', 'happen', 'become',
    'move', 'run', 'hit', 'win', 'lose', 'break', 'open', 'close',
    'add', 'set', 'hold', 'turn', 'drop', 'bring', 'wear', 'buy',
    
    # 情感表达通用词
    'love', 'loved', 'loves', 'like', 'liked', 'likes', 'hate',
    'enjoy', 'proud', 'happy', 'sad', 'angry', 'excited', 'worried',
    
    # 程度和比较词
    'less', 'more', 'most', 'best', 'worst', 'better', 'worse',
    'small', 'big', 'huge', 'little', 'short', 'long', 'high', 'low',
    
    # 从噪音主题2提取的词汇（负面情绪和随机词汇）
    'ugly', 'straight', 'annoy', 'act', 'throw', 'sexy', 'piss', 
    'normal', 'kiss', 'doesnt', 'trans', 'gender', 'scar', 'hat', 
    'dumb', 'delete', 'gf', 'mental',
    
    # 从噪音主题6提取的词汇（自动化系统消息）
    'unfollowed', 'aye', 'item', 'direct',  'closet', 
    'manifest', 'era', 'member',  'join', 'eng', 
     'dozen', 'light',  'tag',
    
    # 从Topic 0提取的词汇（随机表达和填充词）
    'id', 'holy', 'silly', 'anyways', 'forever', 'ik', 'suppose', 
    'fav', 'comment', 'expression', 'idc', 'taste', 'scream', 
    'playful', 'steal', 'explain',  'rly',
    
    # 从Topic 6提取的词汇（多语言混合）
    'noona', 'aku', 'thankyou', 
    'hyung', 'mau', 'bb', 'la', 'bgt', 'yu', 'tu', 
    'banget', 'apa', 'nak', 'dia',
    
    # 从Topic 3提取的词汇（多语言混合）
    'coz', 'ni', 'di', 'bcs', 'receive', 'sana',  'ay', 
    'ano', 'ate',  'ph', 'eh',  'din', 'pero', 
    'cutie', 'beh', 'kita',
    
    # 从Topic 5提取的词汇（随机情感表达）
    'dick',
    
    # 从Topic 13提取的词汇（系统自动化消息）
     'pas', 'ups',  'lee', 
    'capable', 'q', 'comeback', 'streak'
]

if __name__ == "__main__":
    with open("py_Code/custom_stopwords.json","w") as f:
        json.dump(custom_stopwords,f)
