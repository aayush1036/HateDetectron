# HateDetectorBot 

Made by team HateDetectron<br>
```
Aayushmaan Jain (J022)
Pratyush Patro (J047)
Calvin Pinto (J048)
```
with ❤

This bot aims to make discord servers a less toxic place to be in by detecting hate speeches by kicking out the users who send more than 5 hate speech messages. 

If you are in a mood to rant on someone, then make sure to edit the threshold by using the command ```threshold <limit>```

This bot is trained on two sets of data:<br>
1. <a href="https://www.kaggle.com/datasets/mrmorj/hate-speech-and-offensive-language-dataset">Twitter hate speech data</a> for detecting hate speech in messages
2. <a href="https://www.kaggle.com/datasets/parthplc/facebook-hateful-meme-dataset">Facebook Hateful memes dataset</a> to detect hate speech in memes

The jupyter notebooks for reference is also given in the code.

Future prospects:
1. Use Word2Vec embeddings instead of tf-idf
2. Try using transformers like VisualBERT instead of conventional machine learning models


References - <a href="https://arxiv.org/pdf/2012.12975.pdf">Research paper on detecting hate speech in memes</a>