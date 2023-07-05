from urlextract import URLExtract
from wordcloud import WordCloud
import pandas as pd
from collections import Counter
import emoji
import re
import numpy as np
import torch

extract = URLExtract()
def fetch_stats(selected_user,df):
    if(selected_user!='Overall'):
        df=df[df['user']==selected_user]
    num_messages = df.shape[0]
    words = []
    for message in df['message']:
        words.extend(message.split())
    num_media_messages = df[df['message']=='<Media omitted>\n'].shape[0]
    links=[]
    for message in df['message']:
        links.extend(extract.find_urls(message))
    return num_messages, len(words), num_media_messages ,len(links)

def most_busy_users(df):
    x = df['user'].value_counts()
    x = x.head(min(10, len(x)))
    new_df = round((df['user'].value_counts()/df.shape[0])*100,2).reset_index().rename(columns={'user':'name','count':'percent'})
    return x,new_df

def create_wordcloud(selected_user,df):

    f = open('stop_hinglish.txt', 'r')
    stop_words = f.read()

    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]

    temp = df[df['user'] != 'group_notification']
    temp = temp[temp['message'] != '<Media omitted>\n']

    def remove_stop_words(message):
        y = []
        for word in message.lower().split():
            if word not in stop_words:
                y.append(word)
        return " ".join(y)

    wc = WordCloud(width=500,height=500,min_font_size=10,background_color='white')
    temp['message'] = temp['message'].apply(remove_stop_words)
    df_wc = wc.generate(temp['message'].str.cat(sep=" "))
    return df_wc

def most_common_words(selected_user,df):

    f = open('stop_hinglish.txt','r')
    stop_words = f.read()

    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]

    temp = df[df['user'] != 'group_notification']
    temp = temp[temp['message'] != '<Media omitted>\n']

    words = []

    for message in temp['message']:
            for word in message.lower().split():
                if (word not in stop_words):
                    for c in word:
                        if c not in emoji.UNICODE_EMOJI_ENGLISH:
                            words.append(word)
                            break

    most_common_df = pd.DataFrame(Counter(words).most_common(20))
    return most_common_df

def emoji_helper(selected_user,df):
    if (selected_user != 'Overall'):
        df = df[df['user'] == selected_user]
    emojis=[]
    for message in df['message']:
        emojis.extend([c for c in message if c in emoji.UNICODE_EMOJI_ENGLISH])
    emoji_df = pd.DataFrame(Counter(emojis).most_common(len(Counter(emojis))))
    return emoji_df

def monthly_timeline(selected_user,df):
    if (selected_user != 'Overall'):
        df = df[df['user'] == selected_user]

    timeline = df.groupby(['year','month_num','month']).count()['message'].reset_index()
    time=[]
    for i in range(timeline.shape[0]):
        time.append(timeline['month'][i]+"-"+str(timeline['year'][i]))
    timeline['time'] =time
    return timeline

def daily_timeline(selected_user,df):

    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]

    daily_timeline = df.groupby('only_date').count()['message'].reset_index()

    return daily_timeline

def week_activity_map(selected_user,df):
    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]
    return df['day_name'].value_counts()

def month_activity_map(selected_user,df):
    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]
    return df['month'].value_counts()

def activity_heatmap(selected_user,df):

    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]

    user_heatmap = df.pivot_table(index='day_name', columns='period', values='message', aggfunc='count').fillna(0)

    return user_heatmap
def birth_dates(df):
    birthdates = []
    names = []
    for i in range(df.shape[0]):
        msg = df['message'][i].lower()
        if (re.search('happy birthday', msg)):
            if (re.findall('@[A-Za-z0-9]+', df['message'][i])):
                users = re.findall('@[A-Za-z0-9]+', df['message'][i])
                for user in users:
                    if user[1:] not in names:
                        names.append(user[1:])
                        birthdates.append(str(df['month'][i]) + " " + str(df['day'][i]))
    return pd.DataFrame({'contacts':names,'birthdates':birthdates})

def sentiment_analysis(selected_user,df):
    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]
    # sample code
    from transformers import BertTokenizer, BertForSequenceClassification
    tokenizer = BertTokenizer.from_pretrained("ganeshkharad/gk-hinglish-sentiment")
    model = BertForSequenceClassification.from_pretrained("ganeshkharad/gk-hinglish-sentiment")
    if df.shape[0]>600:
        df=df.sample(n=600)

    ans = []
    for i in range(df.shape[0]):
        encoded_input = tokenizer(df['message'].iloc[i], return_tensors='pt')
        output = model(**encoded_input)
        output = np.argmax(output.logits.detach().numpy())
        if (output == 0):
            ans.append('Negative-messages')
        elif (output == 1):
            ans.append('Neutral-messages')
        else:
            ans.append('Positive-messages')

        # output contains 3 lables LABEL_0 = Negative ,LABEL_1 = Nuetral ,LABEL_2 = Positive
    return pd.Series(Counter(ans)),df.shape[0]
