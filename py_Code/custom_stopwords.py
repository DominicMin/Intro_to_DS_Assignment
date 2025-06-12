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
    "red", "blue", "green", "yellow", "white", "purple", 
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


    # Filipino and other non-English vocabulary extensions
    "aku", "hai",  "eid", "bgt", "banget", "kak", "youu", "hru",
    "ilysm", "tt",  "bb", "ure", "bae", "omggg", "pero", "kai",
    "kaya", "goodmorning", "tangina", "wala", "pala", "loml",
    "mau", "apa", "nak", "ini", "ada", "lu", "tu", "dia", "lah", 
    "lagi", "ke", "yang", "udah", "dong", "sama", "que", "unnie", 
    "noona", "oppa",

    # === Stopwords added from LDA topic analysis ===

    # High-frequency stopwords from Topic 14 (noise topic - general expression vocabulary)
    "look", "need", "thank", "today", "bad", "tell", "happy", 
    "morning",  "hate",


    # Internet slang and abbreviations (from Topic 19)
    "smth", "dw", "yk", "bcs", "lh", "wdym", "aye", "ve", "imma", "nigga", "bitch",

    # Single letter and short abbreviations (from Topic 19)
    "st",  "dk", "gose", "oj", "mar", "uni",

    # === Stopwords added from LDA topic analysis ===

    # High-frequency stopwords from Topic 21 (noise topic - general vocabulary)
    # These are the most frequent general vocabulary, which does not contribute to topic modeling
    "like", "know", "wordle", "oh", "good", "think",  "want", 
    "look", "need", "thank", "right", "come", "feel", "work", "watch", 
    "today", "use", "try", "bad", "tell", "happy", "literally", "actually", 
    "cute", "morning", "play", "hate", "sorry", "little","doesnt",
    
    "address", "express", "disbelief", "informal", "amaze", "amazed", "excite", "ready", "appreciate",

    # === Stopwords added from LDA topic analysis (from file lda_22_4704.md) ===
    
    # Topic 4 (noise topic - general daily life vocabulary)
    "drink", "car", "dog", "dad", "room", "food", "sit", "bed", "water", 
    "drive", "throw", "sick", "date",

    # === Stopwords added from topics 1, 9, 18 analysis ===
    
    # From Topic 1 (K-pop related but general social media interactions)
    "random", "single"
    
    # === Stopwords added from LDA topic analysis (lda_19_5260.md) ===

    # High-frequency cross-topic general vocabulary
    
    "recent", "main", "build", "create", "update", "release", "save", "event", 
    "level", "version", "base", "link", "page", "team", "project", "power", 
    "online", "scene", "season", "series", "episode", "cover", "image", "content",
    
    # Social media interaction words
    "edit", "view", "trend", "trending", "reach", "stream", "vote", 
    
    # General descriptive adjectives  
    "gorgeous", "handsome", "adorable", "precious", "cute", "cutest", "soft", 
    "special", "private", "busy", "young", "normal", "silly", "safe",
    
    # General verbs and actions
    "worry", "expect", "choose", "decide", "pick", "catch", "stick", "steal", 
    "shake", "smile", "kiss", "hug", "scream", "sob", "annoy", "bore", "tire",
    
    # General nouns
    "voice", "flower", "summer", "future", "item", "order", "notice", "box", 
    "line", "suit", "cheek", "queen", "king", "flag", "hat", "scar", "mask",
    
    # Internet/gaming terms
    "server", "discord", "chat", "id", "link",  "player", "game", 
    "league", "battle", "model", "seed", "schedule", "general", "term",
    
    # General time and place words
    "past", "instead", "especially", "case", "age", "history", "article", 
    
    
    # General emotional and relationship words
    "relationship", "mental", "emotion", "emotional", "trust", "peace", 
    "energy", "focus",  "pain", "strong", "deep", "grow", "matter",
    
    # General body and appearance words
    "skin", "naked", "tattoo",
    
    # Social media abbreviations and slang
    "lrt", "ep", "vol", "pre", "hv", "pas", "mc", "est", "la", "hr", "lb",
    
    # General connecting words that appear frequently
    "instead", "especially", "seriously", "explain", "mention", "correct", 
    "fair", "force", "stand", "exist", "consider",
    
    # === Stopwords added from LDA topic analysis (lda_20_5029.md) ===
    # Focus on Topics 8, 11, 15, 16
    
    # Topic 8 - Expression and emotion words (noise topic)
    "exclamation", "expression", "discomfort", "awkwardness", "sir", "playful", 
    "blush", "sigh", "tear", "pull", "touch", "bite", "daddy", "mommy", 
    "cuddle", "eager", "hide", "dare", "rock", "neck", "pat", "hole", "bark",
    
    # Topic 11 - General planning and design words (noise topic)  
    "set", "spend", "plan", "design", "color", "email", "draw", "light", 
    "forward", "youtube", "app", "card", "space", "chance", "answer", "test", 
    "paint", "mood", "practice", "final", "gift", "piece", "hopefully", 
    "style", "offer", "easy", "interested", "receive", "list", "challenge", "memory",
    
    # Topic 15 - Social media and personal life words (noise topic)
    "tiktok", "gon", "era", "class", "delete", "piss", "gf", "straight", 
    "obsess", "swear", "suppose", "bf", "sister", "crush", "boyfriend", 
    "sense", "taste", "ruin", "fall", "parent", "pride", "playlist", 
    "cut", "scary", "pop", "lesbian",  "embarrass", "layout", 
     "ab", "giggle",
    
    
    # Additional high-frequency noise words from these topics
    "currently", "hopefully", "forward", "community", "offer", "receive", 
    "interested", "challenge", "memory", "practice",

    # === Stopwords added from LDA topic analysis (lda_19_5687.md) ===
    
    # High-frequency cross-topic general vocabulary that appears in multiple topics
    "photo", "star", "single", "figure",  "important", "interesting", 
    "simple", "example", "form", "attention", "fake", "luck", "tired", "count",
    "available", "expensive", "cheap", "issue", "child", "mother", "father", 
    "public", "mass",   "outfit", "english", 
    "korea", "chapter", "result", "active",
    
    # Music and entertainment related but too general
    "member", "group", "concert", "bias", "solo", "performance", "perform", 
    "artist", "film", "actor", "cast", "news", "party", "dance", "stage", 
    "tour", "ticket",
    
    # Topic 18 noise vocabulary (general meaningless words)
    "beat", "war", "animal", "ball", "wife", "dark", "band", "shot", "lady", 
    "opinion", "sex", "meme", "internet", "american", "original", "club", 
    "evil", "gun", "smoke", "earth", "cringe", "rip", "legend", "dick", 
    "monster", "google", "mid",
    
    # General descriptive and action words that lack specificity
    "suddenly", "fit", "death", "ignore", "fast", "cold", "favourite", "train",
    
    # === Stopwords added from LDA topic analysis (lda_24_5128.md) ===
    
    # K-pop and entertainment related but too general across topics
    "mv", "comeback", "concept", "debut", "kpop", "idol", "cb", "teaser", "track", 
    "choice", "stalker", "devote", "fandom", "vlive", "pc", "bias",
    
    # Social media and interaction words that appear across topics
    "remind", "besties", "circle", "user", "interaction", "profile", "instagram", 
    "code", "click", "everyday", "min", "number", "thread",
    
    # General emotional and descriptive words
    "felt", "remind", "realise", "immediately", "actual", "treat", "lucky", 
    "healthy", "fashion", "weather", "daily", "quality", "contact", "highlight",
    
    # Entertainment and media related but too general
    "netflix", "vibe", "iconic", "plot", "stress", "tv", "lyric", "alive", 
    "survive", "fear", "attack", "mess", "wild", "jealous", "relate",
    
    # Topic 14 noise vocabulary (daily life general words)
    "coffee", "clean", "store", "couple", "shirt", "sell", "tea", "shop", "hang", 
    "glass", "rain", "fly", "ice", "air", "smell", "nap", "sort", "fix", "trip", 
    "hop", "dress", "treat", "extra", "floor", "size", "bag", "cook", "roll", 
    "worth", "quick", "cup", "wall", "deal", "burn", "screen", "cream",
    
    # General action and state words
    "visit", "ride", "interview", "appear", "serve", "jump", "drag", "convince", 
    "lead", "slow", "market", "company", "price", "service", "rate", "rule",
    
    # Color and appearance words that are too general
    "pink", "black", "dress",
    
    # === Stopwords added from LDA topic analysis (lda_19_5222.md) ===
    
    # Cross-topic general action and state words
    "act", "note", "text", "step", "possible", "arrive", "prepare", "protect", 
    "include", "round", "claim", "upset", "marry", "heard", "accidentally", 
    "clothes", "fell",  "boring", "skip", "recover",
    
    # Business and academic related but too general
    "process", "review", "solution", "blog", "response", "website", "personal", 
    "creative", "value", "meeting", "writing", "manage", "journey", "excellent", 
    "current", "major", "podcast", "recent", "global",
    
    # Psychological and social interaction words that are too general
    "truth", "accept", "allow", "situation", "control", "certain", "difficult", 
    "conversation", "teach", "reality", "respect", "mistake", "health", "anxiety",
    
    # Entertainment and media related but too general
    "sing", "fave", "spoiler", "famous", "trailer", "rewatch", "icon", "archive", 
    "anniversary", "upload", "nickname", "record", "menu", "preview",
    
    # Social media and technology related but too general
    "weverse", "spotify", "retweet", "mint", "participate", "carrd", "nft",
    
    # General descriptive words that lack specificity
    "smart", "similar", "double", "incredible", "personally", "impression", 
    "leader", "goal", "darling",
    
    # Topic 15 noise vocabulary (daily life general words)
    "covid", "city", "husband", "park", "flight", "gym", 
    "office", "beach", "lunch", "bar", "travel", "town", "doctor", 
    "door", "pack", "apartment", "pass", "living", "daughter", "street", "son", "dad","mom",
    "father", "mother",
    "shoe", "hotel", "wine", "afraid"
]
with open("py_Code/custom_stopwords.json","w") as f:
    json.dump(custom_stopwords,f)