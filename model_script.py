import pandas as pd
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from sklearn.linear_model import LogisticRegression
import argparse
import pickle

parser = argparse.ArgumentParser()
parser.add_argument('-l', '--url_len', default='0')
parser.add_argument('-t', '--tld', default='com')
parser.add_argument('-a', '--https', default='0')
args = parser.parse_args()

URL_LEN = int(args.url_len)
TLD = args.tld
HTTPS = int(args.https)

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

logisticRegr = pickle.load(open("./finalized_model.sav", 'rb'))

x_test = pd.DataFrame(columns = ['url_len', 'tld', 'https'])
x_test.loc[len(x_test.index)] = [URL_LEN, TLD, HTTPS]
x_test['tld'] = OrdinalEncoder().fit_transform(x_test.tld.values.reshape(-1,1))
x_test['https'] = OrdinalEncoder().fit_transform(x_test.https.values.reshape(-1,1))

y_pred = logisticRegr.predict(x_test)
if int(y_pred[0]) == 1:
    print("SAFE")
else:
    print("UNSAFE")