from nltk.stem import WordNetLemmatizer, SnowballStemmer, PorterStemmer, LancasterStemmer
from string import punctuation
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords
from string import punctuation
from sklearn.preprocessing import LabelEncoder
import re 
import pandas as pd 

class Preprocess:
    def __init__(self,method='WordNetLemmatizer') -> None:
        self.method = method 
        self.stemmers = {
            'WordNetLemmatizer':WordNetLemmatizer(),
            'SnowballStemmer':SnowballStemmer(language='english'),
            'PorterStemmer':PorterStemmer(),
            'LancesterStemmer':LancasterStemmer()
        }
        self.stemmer = self.stemmers[self.method]
        self.stopWords = list(stopwords.words('english')) + list(punctuation)
        self.vectorizer = TfidfVectorizer()
        self.encoder = LabelEncoder()
    
    def preprocess(self,message):
        message = message.lower()
        #Remove links 
        message = re.sub('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+#]|[!*\(\),]|'\
                        '(?:%[0-9a-fA-F][0-9a-fA-F]))+','', message)
        # Remove extra spaces 
        message = re.sub(' +', ' ', message)
        # Remove mentions 
        message =re.sub("(@[A-Za-z0-9_]+)","", message)
        # Remove all non alphanumeric characters 
        message = re.sub("^[A-Za-z0-9_-]*$", "", message)
        # Remove Emojis 
        emoji_pattern = re.compile(
            "["
            u"\U0001F600-\U0001F64F"  # emoticons
            u"\U0001F300-\U0001F5FF"  # symbols & pictographs
            u"\U0001F680-\U0001F6FF"  # transport & map symbols
            u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
            u"\U00002702-\U000027B0"
            u"\U000024C2-\U0001F251"
            "]+",
            flags=re.UNICODE,
        )
        message = emoji_pattern.sub('',message)
        
        if self.method == 'WordNetLemmatizer':
            message = ' '.join([self.stemmer.lemmatize(word) for word in message.split() if word not in self.stopWords])
        else:
            message = ' '.join([self.stemmer.stem(word) for word in message.split() if word not in self.stopWords])
        return message

    def fit(self,X,y):
        X = X.apply(self.preprocess)
        self.vectorizer.fit(X)
        if y is not None:
            self.encoder.fit(y)

    def transform(self,X,y=None):
        if isinstance(X,pd.Series):
            X = X.apply(self.preprocess)
            X = self.vectorizer.transform(X)
            X = X.toarray()
        else:
            X = self.preprocess(X)
            X = self.vectorizer.transform([X])
            X = X.toarray()
        if y is not None:
            y = self.encoder.transform(y)
            return X,y
        else:
            return X
    
    def fit_transform(self,X,y):
        self.fit(X,y)
        X,y = self.transform(X,y)
        return X,y
    
    def predictNew(self, message,model):
        message = self.transform(message)
        pred = model.predict(message)
        return self.encoder.inverse_transform(pred)[0]