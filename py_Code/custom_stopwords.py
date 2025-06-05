import json

'''
We choose to first list the stopwords in Python list and then convert it to JSON file.
That is because Python file allows comments and is more readable, so that 
we can categorize the stopwords and make it more organized.
'''

custom_stopwords=[
    # Single letter words
    "l", "v", "g", "x", "q",
    
    # Generic degree adverbs
    "autometically","unfollow","extremely", "super", "really", "very", "totally", 
    "genuinely", "completely", "absolutely", 
    "perfectly", "exactly", "definitely", "obviously", "clearly", "basically", 
    "literally", "actually", "generally", "usually", "normally", "simply", 
    "truly", "fully", "easily", "barely", "hardly", "recently", "lately",
    
    # Interjections and exclamatory words
    "ah", "oh", "uh", "um", "huh", "yo", "ya", "yay", "aww", "awww", "aw", 
    "ahhh", "ahh", "ohhh", "ohh", "hehe", "hiii", "hii", "bye", "yea", "yep", 
    "nah", "dude", "bro", "bestie", "babe", "dear", "honey", "sweetie",
    
    # Internet slang abbreviations
    "gt", "ll", "tf", "mf", "idc", "istg", "rlly", "rly", "ik", "omfg", 
    "lmfao", "lmaooo", "lmaoo", "xd", "uwu", "oomf", "twt", "rt", "dm", 
    "pm", "kst", "bbl", "gm", "gn", "acc", "pfp", "priv", "mutuals",
    
    # Filler words and conjunctions
    "like", "just", "well", "so", "but", "and", "or", "the", "a", "an", 
    "this", "that", "these", "those", "it", "its", "they", "them", "their",
    "he", "she", "his", "her", "we", "us", "our", "you", "your", "i", "me", "my",
    
    # Time-related general words
    "today", "yesterday", "tomorrow", "tonight", "weekend", 
    "monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday",
    "morning", "afternoon", "evening", "night", "early", "late", "later",
    
    # General emotional words
    "good", "bad", "nice", "great", "amazing", "awesome", "perfect", "beautiful",
    "lovely", "sweet", "cute", "cool", "hot", "sexy", "ugly", "stupid", "dumb",
    "crazy", "insane", "mad", "angry", "sad", "happy", "excited", "proud",
    
    # General body parts
    "head", "face", "eye", "nose", "mouth", "lip", "ear", "hair", "hand", 
    "finger", "arm", "leg", "foot", "body", "heart", "brain",
    
    # General color words
    "red", "blue", "green", "yellow", "black", "white", "pink", "purple", 
    "orange", "brown", "gray", "grey",
    
    # Quantity and degree words
    "much", "many", "more", "most", "less", "least", "few", "little", "big", 
    "small", "large", "huge", "tiny", "long", "short", "tall", "high", "low",
    "half", "full", "empty", "whole", "entire", "complete",
    
    # General action words
    "go", "come", "get", "put", "take", "give", "make", "do", "say", "tell", 
    "know", "think", "see", "look", "watch", "hear", "listen", "feel", "want", 
    "need", "like", "love", "hate", "try", "use", "work", "play", "run", "walk",
    
    # Location words
    "here", "there", "where", "up", "down", "in", "out", "on", "off", "over", 
    "under", "above", "below", "inside", "outside", "left", "right", "front", 
    "back", "top", "bottom", "side", "middle", "around", "near", "far",
    
    # Question words and pronouns
    "what", "when", "where", "why", "how", "who", "which", "whose", "whom",
    "something", "anything", "nothing", "everything", "someone", "anyone", 
    "everyone", "nobody", "somebody", "anybody", "everybody",
    
    # Common meaningless words in social media
    "moot", "stan", "stans", "slay", "periodt", "bestie", "bestie", "congrats", 
    "congratulation", "welcome", "thanks", "thank", "please", "sorry", "excuse",
    
    # Despite manually adding the stopwords, we also use LLM to scan
    # and add the stopwords that are not included in the list.
    # === Stopwords added from LDA topic analysis ===

    # Topic 18 (Non-English Vocabulary)
    "na", "po", "ko", "ang", "ako", "ka", "lang", "mo", "ng", "ba", "sa", 
    "naman", "pa", "huhu", "si", "talaga", "yung", "yan", "pogi", "ano", 
    "ate", "di", "sana", "ni", "manifest", "ay", "cutie", "hindi", "beh",
    
    
    # === Stopwords added from CSV file (1001-2000 rows) ===
    
    # Degree adverbs and emphasis words (from CSV scanning)
    "nearly", "quickly", "sadly", "usual", "incredibly", "slightly", "properly", 
    "certainly", "possibly", "specifically", "highly", "physically", "overly", 
    "entirely", "mostly", "mainly", "directly", "rarely", "plenty", "surely",
    
    # Interjections and exclamatory words (from CSV scanning)
    "ahhhh", "aaaa", "aaa", "aaaaa", "noo", "nooo", "noooo", "nooooo", "ooo", 
    "oooh", "ooh", "oops", "hmmm", "uhm", "hehehe", "omgg", "omgggg", "ew", 
    "ayo", "naur", "alr", "dang", "heck", "aint", "lmaoooo", "lmfaooo", "sooooo", 
    "youuuu", "youuu", "gooo", "goooo", "yasss", "yikes",
    
    # Single letter and short interjections (from CSV scanning)
    "ca",  "ho", "ki", "im", "z", "r", "ai", "en", "cr", "se", "le", 
    "wi", "ia", "ne", "ch", "er", "ye", "hm", "ft", "fb", "tw", "yt", "ny", 
    "pr", "ac", "oc", "pt", "ta", "wo", "ao", "ga", "bu",  "esp", "ish",
    
    # Internet slang and abbreviations (from CSV scanning)
    "urself", "theres", "arent", 
    "tryna", "coz", "bout", "asf", "jus", "lik", "hav", "fam", "favs",
    
    # === Stopwords added from CSV file (2001-3000 rows) ===
    
    # Degree adverbs and emphasis words (from CSV 2001-3000 rows)
    "mainly", "somewhat", "instantly", "naturally", "necessarily", "desperately", 
    "casually", "regularly", "actively", "thankfully", "prolly", "apparently",
    
    # Interjections and exclamatory words (from CSV 2001-3000 rows)
    "ahhhhh", "aaaaaa", "ohmygod", "oml", "noooooo", "mmm", "umm", "helloo", 
    "heyy", "hihi", "heyyy", "lmfaoooo", "lmfaoo", "cmon", "oooo", "meh", 
    "whoa", "yooo", "woooo", "wooo", "yayyy", "yayy", "yknow", "yassss", 
    "yeahh", "uhhh", "uhh", "welp", "heh", "hala", "hmmmm", "meeee", "hun",
    
    # Single letter and short abbreviations (from CSV 2001-3000 rows)
    "ph", "bo", "oo", "dd", "va", "uu", "ge", "rm", "mu", "bg", "mx", "sp", 
    "ju", "eu", "sc", "tr", "sk", "mm", "aa", "ou", "je", "ol", "cw", "nga", "mga",
    
    # === Stopwords added from CSV file (3001-4000 rows) ===
    
    # Degree adverbs and emphasis words (from CSV 3001-4000 rows)
    "originally", "luckily", "surprisingly", "ironically", "particularly", 
    "secretly", "necessarily", "consistently", "publicly", "gladly", "fairly", 
    "loudly", "nicely", "heavily", "happily", "differently",
    
    # Interjections and exclamatory words (from CSV 3001-4000 rows)
    "aaaaaaa", "aaaaaaaa", "ooooh", "ummm", "plsss", "helppp", "helpp", "whew", 
    "wat", "heheh", "meeeee", "cuteeee", "cutee", "sameee", "tooo", "tyyy", 
    "tyy", "yayyyy", "yeahhh", "lmfaooooo", "hahahahah", "slayyy", "yah", 
    "huhuhu", "goooooo", "ooooo", "aaah", "uuuu", "frfr", "okie", "oki", 
    "yess", "tru", "helloooo", "hiiiii", "gurl", "girlies", "girlie", "hoy",
    
    # Single letter and short abbreviations (from CSV 3001-4000 rows)
    "vi", "sf", "ap", "thr", "val", "pl", "ff", "bp", "whe", "ci", "del", 
    "ow", "ly", "sg", "pd", "wor", "jp", "hc", "ck", "ww", "ri", "ii", 
    "gl", "fl", "fc", "dp", "ja", "br", "aq", "ml", "kr", "ea", "chi", "gu", 
    "ser", "isa", "sta", "dah", "mag", "tak", "bec", "kasi", "tama", "gago", 
    
    # === Stopwords added from CSV file (4001-end) ===
    
    # Degree adverbs and emphasis words (from CSV 4001-end)
    "practically", "occasionally", "frequently", "overwhelming", "unbelievably", 
    "legitimately", "essentially", "significantly", "successfully", "weirdly", 
    "efficiently", "permanently", "thoughtful", "potentially", "ultimately", 
    "repeatedly", "initially", "objectively", "intentionally", "purposely",
    
    # Interjections and exclamatory words (from CSV 4001-end)
    "ohhhhh", "nooooooo", "okayy", "yessssss", "aha", "cus", "obv", "probs", 
    "wooooo", "yoooo", "yooooo", "toooo", "plssss", "tooooo", "lmaooooo", 
    "lmaoooooo", "hahahha", "hahahahahah", "hehehe", "hehehehe", "ahahaha", 
    "naurrr", "nahh", "nahhh", "wat", "dat", "frr", "frrr", "sht", "phew", 
    "whyyy", "whyy", "cuteeeee", "cuteeee", "sameee", "tryy", "tyyy", "tyty",
    
    # Single letter and short abbreviations (from CSV 4001-end)
    "om", "mp", "gb", "nu", "ag", "pp", "ps", "bs", "ss", "hq", "qu", "es", 
    "mt", "mb", "ei", "ut", "tn", "cl", "exp", "io", "jr", "md", "ms", "sl", 
    "che", "par", "cos", "str", "fin", "ref", "rec", "lf", "vc", "hw", "gp", 
    "sf", "liv", "amo", "peo", "shi", "gw", "nya", "pu", "mak", "bub", "sht", 
    "kan", "ent", "pla", "loo", "whi", "hap", "mot", "teh", "par", "yew", "ddd",

    # === Stopwords added from LDA topic analysis ===

    # High-frequency stopwords from Topic 15 (noise topic - general expression vocabulary)
    "know", "think", "feel", "come", "right", "tell", "watch", "use", "try", 
    "sorry", "little", "long",

    # Stopwords from Topic 11 (noise topic - automation features and scattered content)
    "automatically", "unfollowed", "comment", "wonderful", "direct", "return", 
    "opportunity", "surprise", "dozen", "tag", "join", "bless", "sign", "alright",

    

    # General vocabulary from social media interactions
    "yall", "shut", "reply", "block", "direct", "follower", "mutual", "interact", 
    "goodnight", "forever", "suck", "dead", "ship", "fav", "anyways",

    # Single letter and meaningless abbreviations
    "j", "e", "f", "nd", "sh", "mi", "eng",

    # Overly general vocabulary from explainable topic (conservative addition)
    "express", "address", "disbelief", "informal", "throw", "fall", "tire",
    "smile", "especially", "worry", "choose", "amaze", "piece", "light", 
    "answer", "decide", "color", "matter", "sense", "issue", "control", 
    "case", "mood", "plan", "version", "perform", "voice", "cover", "era", 


    # Filipino and other non-English vocabulary extensions
    "aku", "hai",  "eid", "bgt", "banget", "kak", "youu", "hru", 
    "ilysm", "tt",  "bb", "ure", "bae", "omggg", "pero", "kai", 
    "kaya", "goodmorning", "tangina", "wala", "pala", "loml",
 
    # === Stopwords added from LDA topic analysis ===

    # High-frequency stopwords from Topic 14 (noise topic - general expression vocabulary)
    "look", "need", "thank", "today", "bad", "tell", "happy", 
    "morning",  "hate",

    # Stopwords from Topic 18 (noise topic - abstract concepts and business terms)
    "experience", "self", "create", "human", "thread", "business", "base", 
    "team", "community",  "build", "market", 
    "focus", "company", "level", "project", "power",  "energy", "term", 
    "article", "value",  "future", "opinion",  "grow", "fear",

    

    # Internet slang and abbreviations (from Topic 19)
    "smth", "dw", "yk", "bcs", "lh", "wdym", "aye", "ve", "imma", "nigga", "bitch",

    # Single letter and short abbreviations (from Topic 19)
    "st",  "dk", "gose", "oj", "mar", "uni",

    # === Stopwords added from LDA topic analysis ===

    # High-frequency stopwords from Topic 21 (noise topic - general vocabulary)
    # These are the most frequent general vocabulary, which does not contribute to topic modeling
    "like", "know", "wordle", "oh", "good", "think", "love", "want", 
    "look", "need", "thank", "right", "come", "feel", "work", "watch", 
    "today", "use", "try", "bad", "tell", "happy", "literally", "actually", 
    "cute", "morning", "play", "hate", "sorry", "little","doesnt",
    
    # === Stopwords added from LDA topic analysis ===
    
    # # Vocabulary from Topic 14 (noise topic - mixed daily topics)
    # "child", "date",  "act", "parent", "sick", 
    # "father",  "dad", "stand", "mother", "mental",  
    # "annoy", "age", "lord", "ex", "marry", "sister", "husband", 
    #  "normal", "male", "explain", "instead", "pain", 
    # "room", "hat", "respect", "dog", "lady", "sit", "treat", "son",
    
    # Additional vocabulary from Topic 21 (noise topic - general high-frequency vocabulary)
    "item", "big", "hair", "hear", "head", "cool", "thanks", "nice", "great",
    
    # Overly general vocabulary from explainable topic (conservative addition)
    
    # General vocabulary from Topic 2 (emotion expression)
    "ready", "safe", "warm", "catch", "felt", 
    
    # General vocabulary from Topic 1 (British life)
    "sick", "car", "train",
    
    # General vocabulary from Topic 20 (daily life)
    "drive", "sit", "clean", "cut", "fast",
    
    # General vocabulary from Topic 3 (LGBTQ+ identity)
    "edit", "delete", "user", "main"
]

with open("py_Code/custom_stopwords.json","w") as f:
    json.dump(custom_stopwords,f)