import os
import discord 
import json
import joblib
import logging
import cv2 
import pytesseract

pytesseract.pytesseract.tesseract_cmd = r'C:\Users\jain_\AppData\Local\Programs\Tesseract-OCR\tesseract.exe'

LOG_FORMAT = '%(levelname)s %(asctime)s - %(message)s'
SAVE_PATH = 'attachments/'
logging.basicConfig(filename='logs.log',
                    level=logging.DEBUG, format=LOG_FORMAT)

with open('config.json','rb') as f:
    config = json.load(f)
    logging.info('Loaded the configuration file')

token = config.get('token')
intents = discord.Intents.all()
intents.members = True 
intents.bans = True
client = discord.Client(intents=intents)

with open('preprocess/imagePreprocess.pkl','rb') as f:
    image_preprocess = joblib.load(f)
    logging.info('Loaded preprocessor for image files')

with open('preprocess/textPreprocess.pkl','rb') as f:
    text_preprocess = joblib.load(f)
    logging.info('Loaded preprocessor for text files')

with open('models/imageModel.pkl','rb') as f:
    image_model = joblib.load(f)
    logging.info('Loaded model for image files')

with open('models/textModel.pkl','rb') as f:
    text_model = joblib.load(f)
    logging.info('Loaded model for text files')

if not os.path.exists(SAVE_PATH):
    os.makedirs(SAVE_PATH)
    logging.info('Made path for saving attachments')

hate_counts = {}

class Bot:
    def __init__(self) -> None:
        self.threshold = 5
        self.server_name = None
        self.channel_name = None

bot = Bot()
@client.event 
async def on_ready():
    for guild in client.guilds:
        if guild.name == 'hatespeech-server':
            bot.server_name = guild.name
            names = [member.name for member in guild.members]
            for name in names:
                hate_counts[name] = 0
            channels = guild.channels
            for channel in channels:
                if channel.name == 'general':
                    bot.channel_name = channel.name
                    channel_id = channel.id 
            message =  f"The limit for sending hate messages is {bot.threshold} which can be changed by using '$threshold <limit>' command"
            await client.get_channel(channel_id).send(message)

@client.event
async def on_member_join(member):
    hate_counts[member.name] = 0

@client.event
async def on_message(message):
    if message.author == client.user:
        pass 
    else:
        content = message.content
        if content.startswith('$threshold'):
            try:
                threshold = int(content.split('$threshold')[1].strip())
                bot.threshold = threshold
            except Exception as e:
                await message.channel.send("Usage '$threshold <limit>'")
        pred = text_preprocess.predictNew(content, text_model)
        if pred == 'Hate Speech':
            user = message.author.name 
            hate_counts[user] += 1 
            if hate_counts[user] >=bot.threshold:
                reason = f'Sent more than {bot.threshold} hate speech messages'
                await message.author.send(f'Kicked from {bot.channel_name} of {bot.server_name} for {reason}')
                await message.author.kick(reason=reason)
        if message.attachments != []:
            attachments = message.attachments
            filenames = []
            for attachment in attachments:
                filenames.append(attachment.filename)
                await attachment.save(os.path.join(SAVE_PATH, attachment.filename))
            for filename in filenames:
                img = cv2.imread(os.path.join(SAVE_PATH, filename))
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                text = pytesseract.image_to_string(img)
                pred = image_preprocess.predictNew(text, image_model)
                print(pred)
                print(hate_counts)
        if pred == 'Hate Speech':
            user = message.author.name 
            hate_counts[user] += 1 
            if hate_counts[user] >=bot.threshold:
                reason = f'Sent more than {bot.threshold} hate speech messages'
                await message.author.send(f'Kicked from {bot.channel_name} of {bot.server_name} for {reason}')
                await message.author.kick(reason=reason)
client.run(token)