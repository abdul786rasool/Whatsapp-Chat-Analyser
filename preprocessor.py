import re
import pandas as pd
def preprocess (data):
    pattern = '\d\d/\d\d/\d\d,\s[0-9]+:\d\d\s[a-z]m\s-\s'
    messages = re.split(pattern, data)
    messages = messages[1:]
    dates = re.findall(pattern, data)
    for i in range(len(dates)):
        dates[i] = re.sub('am', 'AM', dates[i])
        dates[i] = re.sub('pm', 'PM', dates[i])
    df = pd.DataFrame({'user_message': messages, 'message-date': dates})
    df['message-date'] = pd.to_datetime(df['message-date'], format="%d/%m/%y, %I:%M %p - ")
    df.rename(columns={'message-date': 'date'}, inplace=True)

    # separate users and messages
    users = []
    messages = []
    for message in df['user_message']:
        entry = re.split(':\s', message)
        if entry[1:]:
            users.append(entry[0])
            messages.append(entry[1])
        else:
            users.append('group_notification')
            messages.append(entry[0])
    df['user'] = users
    df['message'] = messages
    df.drop(columns=['user_message'], inplace=True)

    df['year'] = df['date'].dt.year
    df['month'] = df['date'].dt.month_name()
    df['month_num'] = df['date'].dt.month
    df['only_date']=df['date'].dt.date
    df['day'] = df['date'].dt.day
    df['day_name']=df['date'].dt.day_name()
    df['hour'] = df['date'].dt.hour
    df['minute'] = df['date'].dt.minute

    period = []
    for hour in df[['day_name', 'hour']]['hour']:
        if hour == 23:
            period.append(str(hour) + "-" + str('00'))
        elif hour == 0:
            period.append(str('00') + "-" + str(hour + 1))
        else:
            period.append(str(hour) + "-" + str(hour + 1))

    df['period'] = period

    return df