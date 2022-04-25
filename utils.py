from unittest import result
from nltk.stem import WordNetLemmatizer, SnowballStemmer, PorterStemmer, LancasterStemmer
from string import punctuation
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords
from string import punctuation
from sklearn.preprocessing import LabelEncoder
import re 
import pandas as pd 
import numpy as np 
import easyocr
class Preprocess:
    def __init__(self,method='WordNetLemmatizer') -> None:
        """Preprocesses the text for the machine learning model

        Args:
            method (str, optional): The method you would like to use for preprocessing. Defaults to 'WordNetLemmatizer'.
        """        
        self.method = method 
        # Create a list of stemmers/lemmatizers available
        self.stemmers = {
            'WordNetLemmatizer':WordNetLemmatizer(),
            'SnowballStemmer':SnowballStemmer(language='english'),
            'PorterStemmer':PorterStemmer(),
            'LancesterStemmer':LancasterStemmer()
        }
        # Assign the chosen stemmer
        self.stemmer = self.stemmers[self.method]
        # Initialize a list of stopwords and punctuation to remove
        self.stopWords = list(stopwords.words('english')) + list(punctuation)
        # Initialize the tf-idf vectorizer
        self.vectorizer = TfidfVectorizer()
        # Initialize the label encoder 
        self.encoder = LabelEncoder()
    
    def preprocess(self,message:str)->str:
        """Preprocesses the given message to make extract useful content and make it suitable for machine learning models

        Args:
            message (str): The message you would like to preprocess

        Returns:
            str: The preprocessed message
        """        
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
            u"\U0001F600-\U0001F64F"  
            u"\U0001F300-\U0001F5FF"  
            u"\U0001F680-\U0001F6FF"  
            u"\U0001F1E0-\U0001F1FF"  
            u"\U00002702-\U000027B0"
            u"\U000024C2-\U0001F251"
            "]+",
            flags=re.UNICODE,
        )
        message = emoji_pattern.sub('',message)
        # Stem/lemmatize the preprocessed text
        if self.method == 'WordNetLemmatizer':
            message = ' '.join([self.stemmer.lemmatize(word) for word in message.split() if word not in self.stopWords])
        else:
            message = ' '.join([self.stemmer.stem(word) for word in message.split() if word not in self.stopWords])
        return message

    def fit(self,X:pd.Series,y:pd.Series)->None:
        """Fits the label encoder and the tf-idf vectorizer

        Args:
            X (pd.Series): The text which you want to convert into tf-idf vectors
            y (pd.Series): The labels which you want to encode
        """        
        # Preprocess the text
        X = X.apply(self.preprocess)
        # Fit the vectorizer
        self.vectorizer.fit(X)
        # Fit the label encoder if labels are provided
        if y is not None:
            self.encoder.fit(y)

    def transform(self,X:pd.Series,y=None)->tuple:
        """Applies the transformation to the given data
        Encodes the messages(X) into a tf-idf vector
        if y is also specified, encodes the lables using label encoding

        Args:
            X (pd.Series): The series/array of messages you would like to transform
            y (pd.Series, optional): The series/array of labels you would like to encode. Defaults to None.

        Returns:
            tuple: The tuple containing the transformed versions of specified inputs
        """        
        # Check if X is a series
        if isinstance(X,pd.Series):
            # Apply the preprocess function to the series
            X = X.apply(self.preprocess)
            # Transform X into tf-idf vectors
            X = self.vectorizer.transform(X)
            # Convert X to an array
            X = X.toarray()
        else:
            # Preprocess X
            X = self.preprocess(X)
            # Transform X into tf-idf vector
            X = self.vectorizer.transform([X])
            # Convert X to an array
            X = X.toarray()
        # Encode the labes if they are specified
        if y is not None:
            y = self.encoder.transform(y)
            return X,y
        else:
            return X
    
    def fit_transform(self,X:pd.Series,y:pd.Series)->tuple:
        """Shortcut for fit and transfroming the data in one function

        Args:
            X (pd.Series): The messages you would like to fit and transform
            y (pd.Series): The labels you would like to fit and transform

        Returns:
            tuple: The tuple containing the tf-idf vector of messages and encodings of the labels 
        """
        # Fit the data         
        self.fit(X,y)
        # Transform using the data
        X,y = self.transform(X,y)
        return X,y
    
    def predictNew(self, message:str,model)->str:
        """Predict on the basis of new message

        Args:
            message (str): The message you would want to predict
            model (Any): The model which you would like to use to make predictions 

        Returns:
            str: The predicted labels 
        """        
        # Transform the message into tf-idf vectors
        message = self.transform(message)
        # Make predictions using the model 
        pred = model.predict(message)
        # Decode the predictions 
        return self.encoder.inverse_transform(pred)[0]

class Bot:
    def __init__(self) -> None:
        """Creates a discord bot configuration instance where you can set the properties of the bot
        """        
        self.threshold = 5 # Warning threshold
        self.server_name = 'hatespeech-server' # your SERVER_NAME here 
        self.channel_name = 'general' # your CHANNEL_NAME here 
        self.allowed_types = ['png','jpeg','jpg']
        self.warning_threshold = np.ceil(self.threshold/2)

reader = easyocr.Reader(lang_list=['en'])

def detect_text(path):
    texts = []
    results = reader.readtext(path,paragraph=False)
    if len(results)>1:
        for result in results:
            text = result[1]
            texts.append(text)
        return ' '.join(texts)
    else:
        return results[0][1]
