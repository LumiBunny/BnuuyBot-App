import re
from enum import Enum

class SentimentStrength(Enum):
    STRONG_NEGATIVE = -2  # hate, despise
    NEGATIVE = -1        # dislike
    NEUTRAL = 0          # neutral/unknown
    POSITIVE = 1         # like
    STRONG_POSITIVE = 2  # love, adore
    FAVOURITE = 3        # favourite

class SentimentAnalyzer:
    def __init__(self):
        # Initialize your patterns here, but now as individual words
        self.favourite_words = r'favorite|favourite|fave|fav|best|top|ultimate'
        self.positive_words = r'like|likes|liking|enjoy|enjoys|enjoying|good|nice|great|pleasant|prefer|prefers|satisfying'
        self.strong_positive_words = r'love|loves|loving|adore|adores|adoring|amazing|fantastic|excellent|outstanding'
        self.negative_words = r'dislike|dislikes|disliking'
        self.strong_negative_words = r'hate|hates|hating|despise|despises|despising|loathe|loathes|loathing'
        
        # Special cases for negative sentiments that are phrases
        self.negative_phrases = [
            (r'\bnot\s+(?:like|enjoy)\b', 'not like'),
            (r'\bdon\'?t\s+(?:like|enjoy)\b', "don't like"),
            (r'\bcan\'?t\s+stand\b', "can't stand")
        ]

        # Patterns that check for negation before sentiment words
        self.sentiment_patterns = {
            SentimentStrength.FAVOURITE: [
                r'\b(?:' + self.favourite_words + r')\b'
            ],
            SentimentStrength.STRONG_POSITIVE: [
                r'\breally\s+(?:' + self.strong_positive_words + r')\b',
                r'\b(?:' + self.strong_positive_words + r')\b',
                r'\bcan\'?t\s+get\s+enough\s+of\b',
            ],
            SentimentStrength.POSITIVE: [
                r'\b(?:' + self.positive_words + r')\b',
                r'\binto\b',
            ],
            SentimentStrength.NEGATIVE: [
                r'\b(?:' + self.negative_words + r')\b',
                r'\bnot\s+(?:a\s+fan\s+of|into)\b',
            ],
            SentimentStrength.STRONG_NEGATIVE: [
                r'\breally\s+(?:' + self.negative_words + r')\b',
                r'\b(?:' + self.strong_negative_words + r')\b',
                r'\bcan\'?t\s+stand\b',
            ]
        }
        
        # Compile patterns
        self.compiled_patterns = {
            strength: re.compile('|'.join(patterns), re.IGNORECASE)
            for strength, patterns in self.sentiment_patterns.items()
        }

    async def conjugate_verb(self, sentiment: str) -> str:
        words = sentiment.split()
        verb = words[-1].lower()  # Convert to lowercase for consistency

        irregular_verbs = {
            'be': 'is',
            'have': 'has',
            'do': 'does',
            'go': 'goes',
            'love': 'loves',
            'hate': 'hates',
            'adore': 'adores',
            'despise': 'despises',
            'loathe': 'loathes',
            'prefer': 'prefers',
            'satisfy': 'satisfies',
            'enjoy': 'enjoys',
            'like': 'likes',
            'dislike': 'dislikes',
            'favorite': 'favourite',
            'favourite': 'favourite',
            'fave': 'fave',
            'fav': 'fav',
            'best': 'best',
            'top': 'top',
            'ultimate': 'ultimate',
            's tier': 's tier'
        }   

        # Check for irregular verbs first
        if verb in irregular_verbs:
            return irregular_verbs[verb]

        # If it's already in third person singular, return as is
        if verb.endswith('s') and not verb.endswith(('ss', 'ch', 'sh', 'x', 'z')):
            return verb

        # Apply regex pattern for regular verbs
        pattern = r'^(.*?)(y|[^aeiou])$'
        match = re.match(pattern, verb)
        if match:
            stem, end = match.groups()
            if end == 'y' and stem[-1] not in 'aeiou':
                return stem + 'ies'
            elif end in ['s', 'ch', 'sh', 'x', 'z']:
                return verb + 'es'
            else:
                return verb + 's'
        
        # If no pattern matched, just add 's'
            return verb + 's'

    async def get_sentiment(self, text: str) -> list[str]:
        text = text.lower()
        sentiments = []
        
        # Check for negative phrases first
        for pattern, phrase in self.negative_phrases:
            if re.search(pattern, text):
                sentiments.append(phrase)
                return sentiments

        # Then check for sentiment patterns
        for strength, pattern in self.compiled_patterns.items():
            matches = pattern.findall(text)
            sentiments.extend(matches)
        
        # If no sentiment is found, return an empty list
        return sentiments if sentiments else []

    async def classify_sentiment_word(self, word: str) -> SentimentStrength:
        word = word.lower()
        
        # Check for negative phrases
        for _, phrase in self.negative_phrases:
            if word == phrase:
                return SentimentStrength.NEGATIVE

        for strength, pattern in self.patterns.items():
            if pattern.search(word):
                return strength
        return SentimentStrength.NEUTRAL

    async def strength_to_word(self, strength: SentimentStrength) -> str:
        if strength == SentimentStrength.FAVOURITE:
            return 'favourite'
        elif strength == SentimentStrength.STRONG_POSITIVE:
            return 'loves'
        elif strength == SentimentStrength.POSITIVE:
            return 'likes'
        elif strength == SentimentStrength.NEGATIVE:
            return 'dislikes'
        elif strength == SentimentStrength.STRONG_NEGATIVE:
            return 'hates'
        else:
            return 'has mentioned'