import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import pytest
from messages import ChatHistory
from topic_classifier import ClassifyTopic
from tests.sentiments import SentimentAnalyzer
from llm_models import LLMModels

@pytest.fixture(scope="module")
async def models():
    models = LLMModels()
    yield models
    await models.cleanup()

@pytest.fixture(scope="module")
async def history():
    history = ChatHistory()
    yield history
    await history.cleanup()

@pytest.fixture
def user_id():
    return "Lumi"

@pytest.mark.asyncio
async def test_classify_food_1(history, models):
    analyzer = SentimentAnalyzer()
    sentence = "I love pizza."
    sentiment = await analyzer.get_sentiment(sentence)
    if sentiment != 'neutral':
        sentiment = analyzer.conjugate_verb(sentiment)
    print(f"Result: {sentiment}")
    assert "loves" in sentiment

@pytest.mark.asyncio
async def test_classify_food_2(history, models):
    analyzer = SentimentAnalyzer()
    sentence = "I dislike chocolate ice cream."
    sentiment = await analyzer.get_sentiment(sentence)
    if sentiment != 'neutral':
        sentiment = analyzer.conjugate_verb(sentiment)
    print(f"Result: {sentiment}")
    assert "dislikes" in sentiment

@pytest.mark.asyncio
async def test_classify_music_1(history, models):
    analyzer = SentimentAnalyzer()
    sentence = "I love classical music."
    sentiment = await analyzer.get_sentiment(sentence)
    if sentiment != 'neutral':
        sentiment = analyzer.conjugate_verb(sentiment)
    print(f"Result: {sentiment}")
    assert "loves" in sentiment

@pytest.mark.asyncio
async def test_classify_music_2(history, models):
    analyzer = SentimentAnalyzer()
    sentence = "I enjoy playing the piano."
    sentiment = await analyzer.get_sentiment(sentence)
    if sentiment != 'neutral':
        sentiment = analyzer.conjugate_verb(sentiment)
    print(f"Result: {sentiment}")
    assert "enjoys" in sentiment


@pytest.mark.asyncio
async def test_classify_hobbies_1(history, models):
    analyzer = SentimentAnalyzer()
    sentence = "I love reading."
    sentiment = await analyzer.get_sentiment(sentence)
    if sentiment != 'neutral':
        sentiment = analyzer.conjugate_verb(sentiment)
    print(f"Result: {sentiment}")
    assert "loves" in sentiment

@pytest.mark.asyncio
async def test_classify_hobbies_2(history, models):
    analyzer = SentimentAnalyzer()
    sentence = "I really dislike knitting."
    sentiment = await analyzer.get_sentiment(sentence)
    if sentiment != 'neutral':
        sentiment = analyzer.conjugate_verb(sentiment)
    print(f"Result: {sentiment}")
    assert "dislikes" in sentiment

@pytest.mark.asyncio
async def test_classify_video_games_1(history, models):
    analyzer = SentimentAnalyzer()
    sentence = "Minecraft is my favorite video game."
    sentiment = await analyzer.get_sentiment(sentence)
    if sentiment != 'neutral':
        sentiment = analyzer.conjugate_verb(sentiment)
    print(f"Result: {sentiment}")
    assert "favorite" in sentiment

@pytest.mark.asyncio
async def test_classify_video_games_2(history, models):
    analyzer = SentimentAnalyzer()
    sentence = "I dislike the Xbox 360."
    sentiment = await analyzer.get_sentiment(sentence)
    if sentiment != 'neutral':
        sentiment = analyzer.conjugate_verb(sentiment)
    print(f"Result: {sentiment}")
    assert "dislikes" in sentiment

@pytest.mark.asyncio
async def test_classify_projects_1(history, models):
    analyzer = SentimentAnalyzer()
    sentence = "I love working on my home renovations."
    sentiment = await analyzer.get_sentiment(sentence)
    if sentiment != 'neutral':
        sentiment = analyzer.conjugate_verb(sentiment)
    print(f"Result: {sentiment}")
    assert "loves" in sentiment

@pytest.mark.asyncio
async def test_classify_projects_2(history, models):
    analyzer = SentimentAnalyzer()
    sentence = "I like working on my personal website."
    sentiment = await analyzer.get_sentiment(sentence)
    if sentiment != 'neutral':
        sentiment = analyzer.conjugate_verb(sentiment)
    print(f"Result: {sentiment}")
    assert "likes" in sentiment

@pytest.mark.asyncio
async def test_classify_streaming_1(history, models):
    analyzer = SentimentAnalyzer()
    sentence = "I love watching Twitch streams."
    sentiment = await analyzer.get_sentiment(sentence)
    if sentiment != 'neutral':
        sentiment = analyzer.conjugate_verb(sentiment)
    print(f"Result: {sentiment}")
    assert "loves" in sentiment

@pytest.mark.asyncio
async def test_classify_streaming_2(history, models):
    analyzer = SentimentAnalyzer()
    sentence = "I like hanging out with chat."
    sentiment = await analyzer.get_sentiment(sentence)
    if sentiment != 'neutral':
        sentiment = analyzer.conjugate_verb(sentiment)
    print(f"Result: {sentiment}")
    assert "likes" in sentiment

@pytest.mark.asyncio
async def test_classify_vtubing_1(history, models):
    analyzer = SentimentAnalyzer()
    sentence = "I love watching vtubers."
    sentiment = await analyzer.get_sentiment(sentence)
    if sentiment != 'neutral':
        sentiment = analyzer.conjugate_verb(sentiment)
    print(f"Result: {sentiment}")
    assert "loves" in sentiment

@pytest.mark.asyncio
async def test_classify_vtubing_2(history, models):
    analyzer = SentimentAnalyzer()
    sentence =  "I am a bunny girl vtuber."
    sentiment = await analyzer.get_sentiment(sentence)
    if sentiment != 'neutral':
        sentiment = analyzer.conjugate_verb(sentiment)
    print(f"Result: {sentiment}")
    assert "neutral" in sentiment

@pytest.mark.asyncio
async def test_classify_cooking_1(history, models):
    analyzer = SentimentAnalyzer()
    sentence =  "I love baking new recipes."
    sentiment = await analyzer.get_sentiment(sentence)
    if sentiment != 'neutral':
        sentiment = analyzer.conjugate_verb(sentiment)
    print(f"Result: {sentiment}")
    assert "loves" in sentiment

@pytest.mark.asyncio
async def test_classify_cooking_2(history, models):
    analyzer = SentimentAnalyzer()
    sentence = "I adore cooking for friends."
    sentiment = await analyzer.get_sentiment(sentence)
    if sentiment != 'neutral':
        sentiment = analyzer.conjugate_verb(sentiment)
    print(f"Result: {sentiment}")
    assert "adores" in sentiment

@pytest.mark.asyncio
async def test_classify_coding_1(history, models):
    analyzer = SentimentAnalyzer()
    sentence = "I love programming in Python."
    sentiment = await analyzer.get_sentiment(sentence)
    if sentiment != 'neutral':
        sentiment = analyzer.conjugate_verb(sentiment)
    print(f"Result: {sentiment}")
    assert "loves" in sentiment

@pytest.mark.asyncio
async def test_classify_coding_2(history, models):
    analyzer = SentimentAnalyzer()
    sentence = "I enjoy solving coding challenges."
    sentiment = await analyzer.get_sentiment(sentence)
    if sentiment != 'neutral':
        sentiment = analyzer.conjugate_verb(sentiment)
    print(f"Result: {sentiment}")
    assert "enjoys" in sentiment

@pytest.mark.asyncio
async def test_classify_content_1(history, models):
    analyzer = SentimentAnalyzer()
    sentence = "I enjoy editing videos."
    sentiment = await analyzer.get_sentiment(sentence)
    if sentiment != 'neutral':
        sentiment = analyzer.conjugate_verb(sentiment)
    print(f"Result: {sentiment}")
    assert "enjoys" in sentiment

@pytest.mark.asyncio
async def test_classify_content_2(history, models):
    analyzer = SentimentAnalyzer()
    sentence = "I like sharing photos on Twitter."
    sentiment = await analyzer.get_sentiment(sentence)
    if sentiment != 'neutral':
        sentiment = analyzer.conjugate_verb(sentiment)
    print(f"Result: {sentiment}")
    assert "likes" in sentiment