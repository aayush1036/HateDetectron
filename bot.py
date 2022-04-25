# Imports 
import os
import discord 
import json
import joblib
import numpy as np 
from utils import detect_text,Bot
# Setting up the path for pytesseract
# Uncomment the line below if you are running it on windows
#pytesseract.pytesseract.tesseract_cmd = r'C:\Users\YOUR_USER_NAME\AppData\Local\Programs\Tesseract-OCR\tesseract.exe'
# Setting up logging 
SAVE_PATH = 'attachments/'
# Loading the configuration file for reading the token
with open('config.json','rb') as f:
    config = json.load(f)

token = config.get('token')
# Setting up permissions for the bot to get the names of members 
intents = discord.Intents.all()
intents.members = True 
# Creating a discord client
client = discord.Client(intents=intents)
# Loading up the preprocessors and models 
with open('preprocess/imagePreprocess.pkl','rb') as f:
    image_preprocess = joblib.load(f)

with open('preprocess/textPreprocess.pkl','rb') as f:
    text_preprocess = joblib.load(f)

with open('models/imageModel.pkl','rb') as f:
    image_model = joblib.load(f)

with open('models/textModel.pkl','rb') as f:
    text_model = joblib.load(f)
# Creating a path to save the images sent in discord chat
if not os.path.exists(SAVE_PATH):
    os.makedirs(SAVE_PATH)
# Initializing a dictionary to store the number of hate speech messages sent by each member 
hate_counts = {}
# Create a basic bot class with the attributes 
bot = Bot()
# Creating a function to be executed when the bot is activated 
@client.event 
async def on_ready():
    for guild in client.guilds:
        # if the bot is in the correct channel
        if guild.name == bot.server_name:
            # Fetch the names of all members 
            names = [member.name for member in guild.members]
            # Initialize the hate speech count to 0 for all members 
            for name in names:
                hate_counts[name] = 0
            # Get the channel to send the initial message to 
            channels = guild.channels
            # Search for the proper channel
            for channel in channels:
                if channel.name == bot.channel_name:
                    # Fetch the channel id 
                    channel_id = channel.id 
            # Frame the message 
            message =  f"The limit for sending hate messages is {bot.threshold} which can be changed by using '$threshold <limit>' command"
            # Send the message in the channel
            await client.get_channel(channel_id).send(message)

# Function to be executed when a new member joins the server
@client.event
async def on_member_join(member):
    # Create an entry in the hate_counts dictionary for the member and initialize the hate speech count to 0
    hate_counts[member.name] = 0

# Function to be executed when a message is sent in the channel
@client.event
async def on_message(message):
    # If the message is from the bot itself, then ignore 
    if message.author == client.user:
        pass 
    else:
        # Fetch the content of the message 
        content = message.content
        # If the message is a threshold command 
        if content.startswith('$threshold'):
            try:
                # if the format is correct, then update the threshold 
                threshold = int(content.split('$threshold')[1].strip())
                bot.threshold = threshold
                await message.channel.send(f'The threshold has been updated to {bot.threshold}')
            except Exception as e:
                # if the format is incorrect, then send the correct usage pattern 
                await message.channel.send("Usage '$threshold <limit>'")
        # If the message is text
        # Make a prediction for that message
        pred = text_preprocess.predictNew(content, text_model)
        # If the message is hate speech 
        if pred == 'Hate Speech':
            # fetch the username of the user 
            user = message.author.name 
            # Increment the hate count 
            hate_counts[user] += 1 
            # If the user exceeds the hate count limit
            if hate_counts[user] == bot.warning_threshold:
                await message.channel.send(f"""
WARNING!! {user} please stop sending hate speech messages, 
you will be removed after sending {int(bot.threshold-bot.warning_threshold)} hate messages""")
            if hate_counts[user] >=bot.threshold:
                # Frame a reason for kicking the user 
                reason = f'Sent more than {bot.threshold} hate speech messages'
                # Send the message before kicking 
                await message.author.send(f'Kicked from {bot.channel_name} of {bot.server_name} for {reason}')
                # Kick the user 
                await message.author.kick(reason=reason)
        # If there is some attachment/image in the message 
        if message.attachments != []:
            # Fetch the attachments 
            attachments = message.attachments
            # Initialize a blank list of filenames 
            filenames = []
            # Loop through all attachments 
            for attachment in attachments:
                # If the attachment is of the allowed file types
                if attachment.filename.split('.')[1].strip() in bot.allowed_types:
                    # Append the filename in the list
                    filenames.append(attachment.filename)
                    # Save the attachment in SAVE_PATH
                    await attachment.save(os.path.join(SAVE_PATH, attachment.filename))
            # Loop through filenames list
            for filename in filenames:
                # Read the image 
                path = os.path.join(SAVE_PATH, filename)
                # Perform OCR to recognize the text in the image
                text = detect_text(path)
                # Make a prediction 
                pred = image_preprocess.predictNew(text, image_model)
                # If the message is hate speech
                if pred == 'Hate Speech':
                    # Get the name of the sender
                    user = message.author.name 
                    # Increment the hate speech for that user 
                    hate_counts[user] += 1 
                    # If the user has exceeded hate speech limit
                    if hate_counts[user] == bot.warning_threshold:
                        await message.channel.send(f"""
WARNING!! {user} please stop sending hate speech messages, 
you will be removed after sending {int(bot.threshold-bot.warning_threshold)} hate messages""")
                    if hate_counts[user] >=bot.threshold:
                        # Frame the reason 
                        reason = f'Sent more than {bot.threshold} hate speech messages'
                        # Send the message before kicking 
                        await message.author.send(f'Kicked from {bot.channel_name} of {bot.server_name} for {reason}')
                        # Kick the user 
                        await message.author.kick(reason=reason)
# Run the code
client.run(token)