import os 
SAVE_PATH = 'attachments/'

files = os.listdir(SAVE_PATH)
for file in files:
    os.remove(os.path.join(SAVE_PATH,file))