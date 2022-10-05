# HateDetectorBot 

Made by team HateDetectron<br>

<a href="mailto:aayushmaan1306@gmail.com">Aayushmaan Jain</a><br>
<a href="mailto:patro.pratyush51@nmims.edu.in">Pratyush Patro</a><br>
<a href="mailto:pinto.calvin52@nmims.edu.in">Calvin Pinto</a>


This bot aims to make discord servers a less toxic place to be in by detecting hate speeches by kicking out the users who send more than 5 hate speech messages. 

If you are in a mood to rant on someone, then make sure to edit the threshold by using the command ```$threshold <limit>```

This bot is trained on two sets of data:<br>
1. <a href="https://www.kaggle.com/datasets/mrmorj/hate-speech-and-offensive-language-dataset">Twitter hate speech data</a> for detecting hate speech in messages
2. <a href="https://www.kaggle.com/datasets/parthplc/facebook-hateful-meme-dataset">Facebook Hateful memes dataset</a> to detect hate speech in memes

The jupyter notebooks for reference is also given in the code.

Future prospects:
1. Use Word2Vec embeddings instead of tf-idf
2. Try using transformers like VisualBERT instead of conventional machine learning models
3. Search for better OCR methods to read meme texts more accuractely

<a href="https://discord.com/api/oauth2/authorize?client_id=953317423723978832&permissions=34822&scope=bot">Try our bot</a>

References - <a href="https://arxiv.org/pdf/2012.12975.pdf">Research paper on detecting hate speech in memes</a>
