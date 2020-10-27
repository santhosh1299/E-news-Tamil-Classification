from django.shortcuts import render
from django.http import HttpResponse
import numpy as np
import pickle
import pandas as pd

from keras.preprocessing.sequence import pad_sequences
from keras.models import   load_model

# Create your views here.
def home(request):
  return(render(request,'nlp_app/home.html'))

def results(request):
    loaded_model = load_model('news_model_local.h5')
    with open('tokenizer.pickle', 'rb') as handle:
      tokenizer = pickle.load(handle)
    MAX_NB_WORDS = 32000
    # Max number of words in each complaint.
    MAX_SEQUENCE_LENGTH = 120
    # This is fixed.
    EMBEDDING_DIM = 100
    news = request.GET['news']
    
    seq = tokenizer.texts_to_sequences(news)
    padded = pad_sequences(seq, maxlen=MAX_SEQUENCE_LENGTH)
    pred = loaded_model.predict(padded)
    labels = ['உலகம்', 'சினிமா', 'தமிழகம்', 'இந்தியா', 'அரசியல்', 'விளையாட்டு']
     
    label = pred, labels[np.argmax(pred[0])]
    print("News Category is: ")
    print(label[1])
    answer = label[1]
    return(render(request,'nlp_app/results.html',{'answer':answer}))