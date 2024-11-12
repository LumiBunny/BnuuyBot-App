import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import pytest
from messages import ChatHistory
from topic_classifier import ClassifyTopic
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
    classifier = ClassifyTopic(models, history)
    await classifier.initialize()  # Initialize the classifier
    sentence = "I love pizza."
    result = await classifier.classify_sentence(sentence)
    print(f"Result: {result}")
    assert "food" in result

@pytest.mark.asyncio
async def test_classify_food_2(history, models):
    classifier = ClassifyTopic(models, history)
    await classifier.initialize()  # Initialize the classifier
    sentence = "I dislike chocolate ice cream."
    result = await classifier.classify_sentence(sentence)
    print(f"Result: {result}")
    assert "food" in result

@pytest.mark.asyncio
async def test_classify_food_3(history, models):
    classifier = ClassifyTopic(models, history)
    await classifier.initialize()  # Initialize the classifier
    history.add("assistant", "Assistant","Oh that sounds good.")
    history.add("user", "Lumi", "I love pizza so much!")
    history.add("assistant", "Assistant", "So do I, with extra cheese.")
    history.add("user", "Lumi", "I love chocolate ice cream too!")
    result = await classifier.classify_list(history, 4)
    print(f"Result: {result}")
    assert "food" in result

@pytest.mark.asyncio
async def test_classify_music_1(history, models):
    classifier = ClassifyTopic(models, history)
    await classifier.initialize()  # Initialize the classifier
    sentence = "I love classical music."
    result = await classifier.classify_sentence(sentence)
    print(f"Result: {result}")
    assert "music" in result

@pytest.mark.asyncio
async def test_classify_music_2(history, models):
    classifier = ClassifyTopic(models, history)
    await classifier.initialize()  # Initialize the classifier
    sentence = "I enjoy playing the piano."
    result = await classifier.classify_sentence(sentence)
    print(f"Result: {result}")
    assert "music" in result

@pytest.mark.asyncio
async def test_classify_music_3(history, models):
    classifier = ClassifyTopic(models, history)
    await classifier.initialize()  # Initialize the classifier
    history.add("user", "Lumi", "I really enjoy listening to music when I work.")
    history.add("assistant", "Assistant", "What are your favourite genres?")
    history.add("user", "Lumi", "Lately I have been listening to EDM and lofi on Spotify.")
    result = await classifier.classify_list(history)
    print(f"Result: {result}")
    assert "music" in result

@pytest.mark.asyncio
async def test_classify_hobbies_1(history, models):
    classifier = ClassifyTopic(models, history)
    await classifier.initialize()  # Initialize the classifier
    sentence = "I love reading."
    result = await classifier.classify_sentence(sentence)
    print(f"Result: {result}")
    assert "hobby" in result

@pytest.mark.asyncio
async def test_classify_hobbies_2(history, models):
    classifier = ClassifyTopic(models, history)
    await classifier.initialize()  # Initialize the classifier
    sentence = "I really dislike knitting."
    result = await classifier.classify_sentence(sentence)
    print(f"Result: {result}")
    assert "hobby" in result

@pytest.mark.asyncio
async def test_classify_hobbies_3(history, models):
    classifier = ClassifyTopic(models, history)
    await classifier.initialize()  # Initialize the classifier
    history.add("user", "Lumi", "I really love knitting and reading books.")
    history.add("assistant", "Assistant", "Oh that sounds fun! It's nice to have ways to enjoy life!")
    history.add("user", "Lumi", "I agree, it's really important to find leisurely activities.")
    result = await classifier.classify_list(history)
    print(f"Result: {result}")
    assert "hobby" in result

@pytest.mark.asyncio
async def test_classify_video_games_1(history, models):
    classifier = ClassifyTopic(models, history)
    await classifier.initialize()  # Initialize the classifier
    sentence = "I love playing Minecraft."
    result = await classifier.classify_sentence(sentence)
    print(f"Result: {result}")
    assert "video games" in result

@pytest.mark.asyncio
async def test_classify_video_games_2(history, models):
    classifier = ClassifyTopic(models, history)
    await classifier.initialize()  # Initialize the classifier
    sentence = "I dislike the Xbox 360."
    result = await classifier.classify_sentence(sentence)
    print(f"Result: {result}")
    assert "video games" in result

@pytest.mark.asyncio
async def test_classify_games_3(history, models):
    classifier = ClassifyTopic(models, history)
    await classifier.initialize()  # Initialize the classifier
    history.add("user", "Lumi", "I was looking at the recent sale on Steam and I saw some stuff I wanted to get.")
    history.add("assistant", "Assistant", "Oh yeah? What piqued your interest?")
    history.add("user", "Lumi", "I saw a game that's been on my wishlist for years at a great price, and some DLC for Cult of the Lamb as well.")
    result = await classifier.classify_list(history)
    print(f"Result: {result}")
    assert "video games" in result

@pytest.mark.asyncio
async def test_classify_projects_1(history, models):
    classifier = ClassifyTopic(models, history)
    await classifier.initialize()  # Initialize the classifier
    sentence = "I love working on my home renovations."
    result = await classifier.classify_sentence(sentence)
    print(f"Result: {result}")
    assert "projects" in result

@pytest.mark.asyncio
async def test_classify_projects_2(history, models):
    classifier = ClassifyTopic(models, history)
    await classifier.initialize()  # Initialize the classifier
    sentence = "I like working on my personal website."
    result = await classifier.classify_sentence(sentence)
    print(f"Result: {result}")
    assert "projects" in result

@pytest.mark.asyncio
async def test_classify_projects_3(history, models):
    classifier = ClassifyTopic(models, history)
    await classifier.initialize()  # Initialize the classifier
    history.add("user", "Assistant", "Did you need me to set a reminder for you?")
    history.add("user", "Lumi", "Yes please! I've been trying to stay on top of things in Jira but I can't seem to stay on track with my current plans.")
    history.add("assistant", "Assistant", "Okay, I will set up a reminder for you.")
    history.add("user", "Lumi", "Thanks! That will help me with scheduling as I work on my current development with this dev project.")
    result = await classifier.classify_list(history, 4)
    print(f"Result: {result}")
    assert "projects" in result

@pytest.mark.asyncio
async def test_classify_streaming_1(history, models):
    classifier = ClassifyTopic(models, history)
    await classifier.initialize()  # Initialize the classifier
    sentence = "I love watching Twitch streams."
    result = await classifier.classify_sentence(sentence)
    print(f"Result: {result}")
    assert "streaming" in result

@pytest.mark.asyncio
async def test_classify_streaming_2(history, models):
    classifier = ClassifyTopic(models, history)
    await classifier.initialize()  # Initialize the classifier
    sentence = "I like hanging out with chat."
    result = await classifier.classify_sentence(sentence)
    print(f"Result: {result}")
    assert "streaming" in result

@pytest.mark.asyncio
async def test_classify_streaming_3(history, models):
    classifier = ClassifyTopic(models, history)
    await classifier.initialize()  # Initialize the classifier
    history.add("user", "Assistant", "Are we going to be streaming today?")
    history.add("user", "Lumi", "I don't think so, I haven't made my streaming schedule yet, but I think I'll be able to set that up for tonight.")
    history.add("assistant", "Assistant", "I hope so, I was looking forward to collabing with you soon.")
    history.add("user", "Lumi", "Don't worry, I already got things set up in OBS for us, and we will go live tommorrow instead.")
    result = await classifier.classify_list(history, 4)
    print(f"Result: {result}")
    assert "streaming" in result

@pytest.mark.asyncio
async def test_classify_vtubing_1(history, models):
    classifier = ClassifyTopic(models, history)
    await classifier.initialize()  # Initialize the classifier
    sentence = "I love watching vtubers."
    result = await classifier.classify_sentence(sentence)
    print(f"Result: {result}")
    assert "vtubing" in result

@pytest.mark.asyncio
async def test_classify_vtubing_2(history, models):
    classifier = ClassifyTopic(models, history)
    await classifier.initialize()  # Initialize the classifier
    sentence =  "I am a bunny girl vtuber."
    result = await classifier.classify_sentence(sentence)
    print(f"Result: {result}")
    assert "vtubing" in result

@pytest.mark.asyncio
async def test_classify_vtubing_3(history, models):
    classifier = ClassifyTopic(models, history)
    await classifier.initialize()  # Initialize the classifier
    history.add("user", "Lumi", "I am so glad I'm an indie vtuber and not part of a big company!")
    history.add("assistant", "Assistant", "Why is that?")
    history.add("user", "Lumi", "I just think there is way too much drama within companies like Niji and Hololive. I'd rather stay away from that!")
    result = await classifier.classify_list(history)
    print(f"Result: {result}")
    assert "vtubing" in result

@pytest.mark.asyncio
async def test_classify_cooking_1(history, models):
    classifier = ClassifyTopic(models, history)
    await classifier.initialize()  # Initialize the classifier
    sentence =  "I love baking new recipes."
    result = await classifier.classify_sentence(sentence)
    print(f"Result: {result}")
    assert "food" in result

@pytest.mark.asyncio
async def test_classify_cooking_2(history, models):
    classifier = ClassifyTopic(models, history)
    await classifier.initialize()  # Initialize the classifier
    sentence = "I enjoy cooking for friends."
    result = await classifier.classify_sentence(sentence)
    print(f"Result: {result}")
    assert "food" in result

@pytest.mark.asyncio
async def test_classify_cooking_3(history, models):
    classifier = ClassifyTopic(models, history)
    await classifier.initialize()  # Initialize the classifier
    history.add("user", "Lumi", "I have been looking at some reels on Instagram with quick and easy recipes, I want to try some out'!")
    history.add("assistant", "Assistant", "Oh that sounds fun! I hope they turn out to be delicious!")
    history.add("user", "Lumi", "I think they will. I am especially looking forward to trying out the vegan options, they sound like they'd be fun to make.")
    result = await classifier.classify_list(history)
    print(f"Result: {result}")
    assert "food" in result

@pytest.mark.asyncio
async def test_classify_coding_1(history, models):
    classifier = ClassifyTopic(models, history)
    await classifier.initialize()  # Initialize the classifier
    sentence = "I love programming in Python."
    result = await classifier.classify_sentence(sentence)
    print(f"Result: {result}")
    assert "coding" in result

@pytest.mark.asyncio
async def test_classify_coding_2(history, models):
    classifier = ClassifyTopic(models, history)
    await classifier.initialize()  # Initialize the classifier
    sentence = "I enjoy solving coding challenges."
    result = await classifier.classify_sentence(sentence)
    print(f"Result: {result}")
    assert "coding" in result

@pytest.mark.asyncio
async def test_classify_coding_3(history, models):
    classifier = ClassifyTopic(models, history)
    await classifier.initialize()  # Initialize the classifier
    history.add("user", "Lumi", "I am stuck on this current problem in my app and I need help!")
    history.add("assistant", "Assistant", "I understand, Python can be complex at times. What did you need help with?")
    history.add("user", "Lumi", "I am just struggling so much with how my functions are structured, there's no way I can push this to GitHub as things are right now.")
    result = await classifier.classify_list(history)
    print(f"Result: {result}")
    assert "coding" in result

@pytest.mark.asyncio
async def test_classify_content_1(history, models):
    classifier = ClassifyTopic(models, history)
    await classifier.initialize()  # Initialize the classifier
    sentence = "I enjoy editing videos."
    result = await classifier.classify_sentence(sentence)
    print(f"Result: {result}")
    assert "social media and content creation" in result

@pytest.mark.asyncio
async def test_classify_content_2(history, models):
    classifier = ClassifyTopic(models, history)
    await classifier.initialize()  # Initialize the classifier
    sentence = "I like sharing photos on Twitter."
    result = await classifier.classify_sentence(sentence)
    print(f"Result: {result}")
    assert "social media and content creation" in result

@pytest.mark.asyncio
async def test_classify_content_3(history, models):
    classifier = ClassifyTopic(models, history)
    await classifier.initialize()  # Initialize the classifier
    history.add("user", "Lumi", "I love creating content and making videos, but I've been feeling a little burnt out recently.")
    history.add("assistant", "Assistant", "That's understandable, making videos for YouTube is a lot of work! It's important to take breaks when you can.")
    history.add("user", "Lumi", "You're right, I can take a step back from videos while I brainstorm ideas and just have fun posting on Twitter and Instagram instead so that my followers don't feel abandoned.")
    result = await classifier.classify_list(history)
    print(f"Result: {result}")
    assert "content" in result