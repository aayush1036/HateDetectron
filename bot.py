import os
import discord 
import json
import joblib
import logging

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
@client.event 
async def on_ready():
    for guild in client.guilds:
        if guild.name == 'hatespeech-server':
            names = [member.name for member in guild.members]
            for name in names:
                hate_counts[name] = 0

@client.event
async def on_member_join(member):
    hate_counts[member.name] = 0
    print(hate_counts)

@client.event
async def on_message(message):
    if message.author == client.user:
        pass 
    else:
        content = message.content
        pred = text_preprocess.predictNew(content, text_model)
        if pred == 'Hate Speech':
            user = message.author.name 
            hate_counts[user] += 1 
            print('Hate Speech')
            if hate_counts[user] >=5:
                print(f'{user} has sent more than 5 hate speech messages')
                reason = 'Sent more than 5 hate speech messages'
                await message.author.kick(reason=reason)
        if message.attachments != []:
            attachments = message.attachments
            for attachment in attachments:
                await attachment.save(os.path.join(SAVE_PATH, attachment.filename))
client.run(token)