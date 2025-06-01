import json
custom_stopwords=[
    # 单字母词汇
    "l", "v", "g", "x", "q",
    
    # 泛用的程度副词
    "autometically","unfollow","extremely", "super", "really", "very", "totally", "genuinely", "completely", "absolutely", 
    "perfectly", "exactly", "definitely", "obviously", "clearly", "basically", 
    "literally", "actually", "generally", "usually", "normally", "simply", 
    "truly", "fully", "easily", "barely", "hardly", "recently", "lately",
    
    # 语气词和感叹词
    "ah", "oh", "uh", "um", "huh", "yo", "ya", "yay", "aww", "awww", "aw", 
    "ahhh", "ahh", "ohhh", "ohh", "hehe", "hiii", "hii", "bye", "yea", "yep", 
    "nah", "dude", "bro", "bestie", "babe", "dear", "honey", "sweetie",
    
    # 网络用语缩写
    "gt", "ll", "tf", "mf", "idc", "istg", "rlly", "rly", "ik", "omfg", 
    "lmfao", "lmaooo", "lmaoo", "xd", "uwu", "oomf", "twt", "rt", "dm", 
    "pm", "kst", "bbl", "gm", "gn", "acc", "pfp", "priv", "mutuals",
    
    # 填充词和连接词
    "like", "just", "well", "so", "but", "and", "or", "the", "a", "an", 
    "this", "that", "these", "those", "it", "its", "they", "them", "their",
    "he", "she", "his", "her", "we", "us", "our", "you", "your", "i", "me", "my",
    
    # 时间相关泛词
    "today", "yesterday", "tomorrow", "tonight", "weekend", 
    "monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday",
    "morning", "afternoon", "evening", "night", "early", "late", "later",
    
    # 情感泛词
    "good", "bad", "nice", "great", "amazing", "awesome", "perfect", "beautiful",
    "lovely", "sweet", "cute", "cool", "hot", "sexy", "ugly", "stupid", "dumb",
    "crazy", "insane", "mad", "angry", "sad", "happy", "excited", "proud",
    
    # 身体部位泛词
    "head", "face", "eye", "nose", "mouth", "lip", "ear", "hair", "hand", 
    "finger", "arm", "leg", "foot", "body", "heart", "brain",
    
    # 颜色词
    "red", "blue", "green", "yellow", "black", "white", "pink", "purple", 
    "orange", "brown", "gray", "grey",
    
    # 数量和程度词
    "much", "many", "more", "most", "less", "least", "few", "little", "big", 
    "small", "large", "huge", "tiny", "long", "short", "tall", "high", "low",
    "half", "full", "empty", "whole", "entire", "complete",
    
    # 动作泛词
    "go", "come", "get", "put", "take", "give", "make", "do", "say", "tell", 
    "know", "think", "see", "look", "watch", "hear", "listen", "feel", "want", 
    "need", "like", "love", "hate", "try", "use", "work", "play", "run", "walk",
    
    # 位置词
    "here", "there", "where", "up", "down", "in", "out", "on", "off", "over", 
    "under", "above", "below", "inside", "outside", "left", "right", "front", 
    "back", "top", "bottom", "side", "middle", "around", "near", "far",
    
    # 疑问词和代词
    "what", "when", "where", "why", "how", "who", "which", "whose", "whom",
    "something", "anything", "nothing", "everything", "someone", "anyone", 
    "everyone", "nobody", "somebody", "anybody", "everybody",
    
    # 社交媒体常见无意义词
    "moot", "stan", "stans", "slay", "periodt", "bestie", "bestie", "congrats", 
    "congratulation", "welcome", "thanks", "thank", "please", "sorry", "excuse",
    
    
    # === 从LDA主题分析中新增的停用词 ===

    # Topic 18 (菲律宾语内容 - 非英语词汇)
    "na", "po", "ko", "ang", "ako", "ka", "lang", "mo", "ng", "ba", "sa", 
    "naman", "pa", "huhu", "si", "talaga", "yung", "yan", "pogi", "ano", 
    "ate", "di", "sana", "ni", "manifest", "ay", "cutie", "hindi", "beh",
    
    
    # === 从CSV文件第1001-2000行中新增的停用词 ===
    
    # 程度副词和强调词 (来自CSV扫描)
    "nearly", "quickly", "sadly", "usual", "incredibly", "slightly", "properly", 
    "certainly", "possibly", "specifically", "highly", "physically", "overly", 
    "entirely", "mostly", "mainly", "directly", "rarely", "plenty", "surely",
    
    # 语气词和感叹词扩展 (来自CSV扫描)
    "ahhhh", "aaaa", "aaa", "aaaaa", "noo", "nooo", "noooo", "nooooo", "ooo", 
    "oooh", "ooh", "oops", "hmmm", "uhm", "hehehe", "omgg", "omgggg", "ew", 
    "ayo", "naur", "alr", "dang", "heck", "aint", "lmaoooo", "lmfaooo", "sooooo", 
    "youuuu", "youuu", "gooo", "goooo", "yasss", "yikes",
    
    # 单字母和短语气词 (来自CSV扫描)
    "ca",  "ho", "ki", "im", "z", "r", "ai", "en", "cr", "se", "le", 
    "wi", "ia", "ne", "ch", "er", "ye", "hm", "ft", "fb", "tw", "yt", "ny", 
    "pr", "ac", "oc", "pt", "ta", "wo", "ao", "ga", "bu",  "esp", "ish",
    
    # 网络用语和缩写扩展 (来自CSV扫描)
    "urself", "theres", "arent", 
    "tryna", "coz", "bout", "asf", "jus", "lik", "hav", "fam", "favs",
    
    # === 从CSV文件第2001-3000行中新增的停用词 ===
    
    # 程度副词和强调词扩展 (来自CSV 2001-3000行)
    "mainly", "somewhat", "instantly", "naturally", "necessarily", "desperately", 
    "casually", "regularly", "actively", "thankfully", "prolly", "apparently",
    
    # 语气词和感叹词进一步扩展 (来自CSV 2001-3000行)
    "ahhhhh", "aaaaaa", "ohmygod", "oml", "noooooo", "mmm", "umm", "helloo", 
    "heyy", "hihi", "heyyy", "lmfaoooo", "lmfaoo", "cmon", "oooo", "meh", 
    "whoa", "yooo", "woooo", "wooo", "yayyy", "yayy", "yknow", "yassss", 
    "yeahh", "uhhh", "uhh", "welp", "heh", "hala", "hmmmm", "meeee", "hun",
    
    # 单字母和短缩写扩展 (来自CSV 2001-3000行)
    "ph", "bo", "oo", "dd", "va", "uu", "ge", "rm", "mu", "bg", "mx", "sp", 
    "ju", "eu", "sc", "tr", "sk", "mm", "aa", "ou", "je", "ol", "cw", "nga", "mga",
    
    # === 从CSV文件第3001-4000行中新增的停用词 ===
    
    # 程度副词和强调词进一步扩展 (来自CSV 3001-4000行)
    "originally", "luckily", "surprisingly", "ironically", "particularly", 
    "secretly", "necessarily", "consistently", "publicly", "gladly", "fairly", 
    "loudly", "nicely", "heavily", "happily", "differently",
    
    # 语气词和感叹词最终扩展 (来自CSV 3001-4000行)
    "aaaaaaa", "aaaaaaaa", "ooooh", "ummm", "plsss", "helppp", "helpp", "whew", 
    "wat", "heheh", "meeeee", "cuteeee", "cutee", "sameee", "tooo", "tyyy", 
    "tyy", "yayyyy", "yeahhh", "lmfaooooo", "hahahahah", "slayyy", "yah", 
    "huhuhu", "goooooo", "ooooo", "aaah", "uuuu", "frfr", "okie", "oki", 
    "yess", "tru", "helloooo", "hiiiii", "gurl", "girlies", "girlie", "hoy",
    
    # 单字母和短缩写最终扩展 (来自CSV 3001-4000行)
    "vi", "sf", "ap", "thr", "val", "pl", "ff", "bp", "whe", "ci", "del", 
    "ow", "ly", "sg", "pd", "wor", "jp", "hc", "ck", "ww", "ri", "ii", 
    "gl", "fl", "fc", "dp", "ja", "br", "aq", "ml", "kr", "ea", "chi", "gu", 
    "ser", "isa", "sta", "dah", "mag", "tak", "bec", "kasi", "tama", "gago", 
    
    # === 从CSV文件第4001行到结尾的新增停用词 ===
    
    # 程度副词和强调词终极扩展 (来自CSV 4001-结尾)
    "practically", "occasionally", "frequently", "overwhelming", "unbelievably", 
    "legitimately", "essentially", "significantly", "successfully", "weirdly", 
    "efficiently", "permanently", "thoughtful", "potentially", "ultimately", 
    "repeatedly", "initially", "objectively", "intentionally", "purposely",
    
    # 语气词和感叹词终极扩展 (来自CSV 4001-结尾)
    "ohhhhh", "nooooooo", "okayy", "yessssss", "aha", "cus", "obv", "probs", 
    "wooooo", "yoooo", "yooooo", "toooo", "plssss", "tooooo", "lmaooooo", 
    "lmaoooooo", "hahahha", "hahahahahah", "hehehe", "hehehehe", "ahahaha", 
    "naurrr", "nahh", "nahhh", "wat", "dat", "frr", "frrr", "sht", "phew", 
    "whyyy", "whyy", "cuteeeee", "cuteeee", "sameee", "tryy", "tyyy", "tyty",
    
    # 单字母和短缩写终极扩展 (来自CSV 4001-结尾)
    "om", "mp", "gb", "nu", "ag", "pp", "ps", "bs", "ss", "hq", "qu", "es", 
    "mt", "mb", "ei", "ut", "tn", "cl", "exp", "io", "jr", "md", "ms", "sl", 
    "che", "par", "cos", "str", "fin", "ref", "rec", "lf", "vc", "hw", "gp", 
    "sf", "liv", "amo", "peo", "shi", "gw", "nya", "pu", "mak", "bub", "sht", 
    "kan", "ent", "pla", "loo", "whi", "hap", "mot", "teh", "par", "yew", "ddd",
    
   
]

with open("py_Code/custom_stopwords_new.json","w") as f:
    json.dump(custom_stopwords,f)