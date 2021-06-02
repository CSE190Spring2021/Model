import pandas as pd
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from sklearn.linear_model import LogisticRegression
import argparse
import pickle
import urllib.request as urllib2
import re
import json
#from urllib2 import urlopen
import tldextract
#import urllib2
from urllib.parse import urlparse, parse_qs
import socket
import requests

parser = argparse.ArgumentParser()
parser.add_argument('-u', '--url', default='https://www.google.com')
#parser.add_argument('-t', '--tld', default='com')
#parser.add_argument('-a', '--https', default='0')
args = parser.parse_args()

URL = args.url
#URL_LEN = int(args.url_len)
#TLD = args.tld
#HTTPS = int(args.https)

"""
train_df = pd.read_csv('./data/train/Webpages_Classification_train_data.csv')
test_df = pd.read_csv('./data/test/Webpages_Classification_test_data.csv')

train_df = train_df[['url', 'url_len', 'ip_add', 'geo_loc', 'tld', 'who_is',
       'https', 'js_len', 'js_obf_len', 'content', 'label']]
test_df = test_df[['url', 'url_len', 'ip_add', 'geo_loc', 'tld', 'who_is',
       'https', 'js_len', 'js_obf_len', 'content', 'label']]

#train_df['geo_loc'] = OrdinalEncoder().fit_transform(train_df.geo_loc.values.reshape(-1,1))
train_df['tld'] = OrdinalEncoder().fit_transform(train_df.tld.values.reshape(-1,1))
#train_df['who_is'] = OrdinalEncoder().fit_transform(train_df.who_is.values.reshape(-1,1))
train_df['https'] = OrdinalEncoder().fit_transform(train_df.https.values.reshape(-1,1))
train_df['label'] = OrdinalEncoder().fit_transform(train_df.label.values.reshape(-1,1))

def parseUrl(s):
    tmp = s.split('://')
    if len(tmp) > 1:
        return (' '.join(s.split('://')[1].strip('www.').replace('.','/').split('/')))
    else:
        return (' '.join(s.split('://')[0].strip('www.').replace('.','/').split('/')))

train_df['url'] = train_df.url.apply(parseUrl)
logisticRegr = LogisticRegression()
#x_train = train_df[['url', 'url_len', 'geo_loc', 'tld', 'who_is',
#       'https', 'js_len']]
x_train = train_df[['url_len', 'tld', 'https']]
y_train = train_df[['label']]
logisticRegr.fit(x_train, y_train)

#test_df['geo_loc'] = OrdinalEncoder().fit_transform(test_df.geo_loc.values.reshape(-1,1))
#test_df['tld'] = OrdinalEncoder().fit_transform(test_df.tld.values.reshape(-1,1))
#test_df['who_is'] = OrdinalEncoder().fit_transform(test_df.who_is.values.reshape(-1,1))
#test_df['https'] = OrdinalEncoder().fit_transform(test_df.https.values.reshape(-1,1))
#test_df['label'] = OrdinalEncoder().fit_transform(test_df.label.values.reshape(-1,1))
#test_df['url'] = test_df.url.apply(parseUrl)

#from sklearn.metrics import confusion_matrix
#from sklearn.metrics import classification_report
# helper method to print basic model metrics
#def metrics(y_true, y_pred):
#    print('Confusion matrix:\n', confusion_matrix(y_true, y_pred))
#    print('\nReport:\n', classification_report(y_true, y_pred))
"""

logisticRegr = pickle.load(open("./Model/finalized_model_new.sav", 'rb'))

x_test = pd.DataFrame(columns = ['url_len', 'geo_loc', 'tld',
       'https'])
URL_LEN = len(URL)
HTTPS = 0
if "https" in URL:
    HTTPS = 1
URL = URL.strip()
if URL[len(URL)-1] == '/':
    URL = URL[:-1]
flag = False
try:
    IP_ADDRESS = socket.gethostbyname(URL[8:])
except:
    print("UNSAFE")
    flag = True

if (flag == False):
    GEO_IP_API_URL  = 'http://ip-api.com/json/' + IP_ADDRESS
    response = urllib2.urlopen(GEO_IP_API_URL)
    data = json.load(response)
    GEO_LOC = data['country']
    ext = tldextract.extract(URL)
    TLD = ext.suffix

    x_test.loc[len(x_test.index)] = [URL_LEN, GEO_LOC, TLD, HTTPS]
    x_test['tld'] = OrdinalEncoder().fit_transform(x_test.tld.values.reshape(-1,1))
    x_test['https'] = OrdinalEncoder().fit_transform(x_test.https.values.reshape(-1,1))
    x_test['geo_loc'] = OrdinalEncoder().fit_transform(x_test.geo_loc.values.reshape(-1,1))

    #x_test = pd.DataFrame(columns = ['url_len', 'tld', 'https'])
    #x_test.loc[len(x_test.index)] = [URL_LEN, TLD, HTTPS]
    #x_test['tld'] = OrdinalEncoder().fit_transform(x_test.tld.values.reshape(-1,1))
    #x_test['https'] = OrdinalEncoder().fit_transform(x_test.https.values.reshape(-1,1))

    y_pred = logisticRegr.predict(x_test)
    if int(y_pred[0]) == 1:
        print("SAFE")
    else:
        print("UNSAFE")
