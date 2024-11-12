import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import pytest
from messages import ChatHistory
from preferences import PreferenceProcessor
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
async def test_classify_multiple_sentences(history, models, user_id):
    test_sentences = [
        "I love listening to classical music.",
        "Pizza is my favorite food.",
        "I enjoy playing video games in my free time.",
        "Coding in Python is my passion.",
        "I'm learning how to draw anime characters."
    ]

    preference_processor = PreferenceProcessor(models, history)
    
    for sentence in test_sentences:
        history.add("user", user_id, sentence)
        result = await preference_processor.process_text(history.get_history(), user_id)
        print(f"\nTest sentence: {sentence}")
        print(f"Result: {result}\n")
        history.get_history().clear()  # Clear history for next test

@pytest.mark.asyncio
async def test_classify_food_1(history, models, user_id):
    history.add("user", user_id, "I love pizza.")
    result = await PreferenceProcessor(models, history).process_text(history.get_history(), user_id)
    print(result)
    assert "Lumi loves pizza" in result

@pytest.mark.asyncio
async def test_classify_food_2(history, models, user_id):
    history.add("user", user_id, "I dislike chocolate ice cream.")
    result = await PreferenceProcessor(models, history).process_text(history.get_history(), user_id)
    print(result)
    assert "Lumi dislikes chocolate ice cream" in result

@pytest.mark.asyncio
async def test_classify_music_1(history, models, user_id):
    history.add("user",user_id, "I love classical music.")
    result = await PreferenceProcessor(models, history).process_text(history.get_history(), user_id)
    print(result)
    assert "Lumi loves classical music" in result

@pytest.mark.asyncio
async def test_classify_music_2(history, models, user_id):
    history.add("user", user_id, "I enjoy playing the piano.")
    result = await PreferenceProcessor(models, history).process_text(history.get_history(), user_id)
    print(result)
    assert "Lumi likes playing the piano" in result

@pytest.mark.asyncio
async def test_classify_hobbies_1(history, models, user_id):
    history.add("user", user_id, "I love reading.")
    result = await PreferenceProcessor(models, history).process_text(history.get_history(), user_id)
    print(result)
    assert "Lumi loves reading" in result

@pytest.mark.asyncio
async def test_classify_hobbies_2(history, models, user_id):
    history.add("user", user_id, "I really dislike knitting.")
    result = await PreferenceProcessor(models, history).process_text(history.get_history(), user_id)
    print(result)
    assert "Lumi hates knitting" in result

@pytest.mark.asyncio
async def test_classify_video_games_1(history, models, user_id):
    history.add("user", user_id, "I love playing Minecraft.")
    result = await PreferenceProcessor(models, history).process_text(history.get_history(), user_id)
    print(result)
    assert "Lumi loves playing Minecraft" in result

@pytest.mark.asyncio
async def test_classify_video_games_2(history, models, user_id):
    history.add("user", user_id, "I dislike the Xbox 360.")
    result = await PreferenceProcessor(models, history).process_text(history.get_history(), user_id)
    print(result)
    assert "Lumi hates the Xbox 360" in result

@pytest.mark.asyncio
async def test_classify_projects_1(history, models, user_id):
    history.add("user", user_id, "I love working on my home renovations.")
    result = await PreferenceProcessor(models, history).process_text(history.get_history(), user_id)
    print(result)
    assert "Lumi loves working on home renovations" in result

@pytest.mark.asyncio
async def test_classify_projects_2(history, models, user_id):
    history.add("user", user_id, "I like working on my personal website.")
    result = await PreferenceProcessor(models, history).process_text(history.get_history(), user_id)
    print(result)
    assert "Lumi likes working on personal websites" in result

@pytest.mark.asyncio
async def test_classify_streaming_1(history, models, user_id):
    history.add("user", user_id, "I love watching Twitch streams.")
    result = await PreferenceProcessor(models, history).process_text(history.get_history(), user_id)
    print(result)
    assert "Lumi loves watching Twitch streams" in result

@pytest.mark.asyncio
async def test_classify_streaming_2(history, models, user_id):
    history.add("user", user_id, "I like hanging out with chat.")
    result = await PreferenceProcessor(models, history).process_text(history.get_history(), user_id)
    print(result)
    assert "Lumi likes hanging out with chat" in result

@pytest.mark.asyncio
async def test_classify_vtubing_1(history, models, user_id):
    history.add("user", user_id, "I love watching vtubers.")
    result = await PreferenceProcessor(models, history).process_text(history.get_history(), user_id)
    print(result)
    assert "Lumi loves watching vtubers" in result

@pytest.mark.asyncio
async def test_classify_vtubing_2(history, models, user_id):
    history.add("user", user_id, "I am a bunny girl vtuber.")
    result = await PreferenceProcessor(models, history).process_text(history.get_history(), user_id)
    print(result)
    assert "Lumi is a vtuber" in result

@pytest.mark.asyncio
async def test_classify_cooking_1(history, models, user_id):
    history.add("user", user_id, "I love baking new recipes.")
    result = await PreferenceProcessor(models, history).process_text(history.get_history(), user_id)
    print(result)
    assert "Lumi loves baking new recipes" in result

@pytest.mark.asyncio
async def test_classify_cooking_2(history, models, user_id):
    history.add("user", user_id, "I enjoy cooking for friends.")
    result = await PreferenceProcessor(models, history).process_text(history.get_history(), user_id)
    print(result)
    assert "Lumi likes cooking for friends" in result

@pytest.mark.asyncio
async def test_classify_coding_1(history, models, user_id):
    history.add("user", user_id, "I love programming in Python.")
    result = await PreferenceProcessor(models, history).process_text(history.get_history(), user_id)
    print(result)
    assert "Lumi loves programming in Python" in result

@pytest.mark.asyncio
async def test_classify_coding_2(history, models, user_id):
    history.add("user", user_id, "I enjoy solving coding challenges.")
    result = await PreferenceProcessor(models, history).process_text(history.get_history(), user_id)
    print(result)
    assert "Lumi likes solving coding challenges" in result