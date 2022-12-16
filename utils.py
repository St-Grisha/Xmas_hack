import math
import pandas as pd

cat_cols = ['paymentSystem', 'providerId', 'bankCountry', 'partyId', 'shopId',
       'currency', 'flag_ip', 'flag_ip_party', 'card_ip', 'card_email',
       'nan_ip', 'nan_email', 'popular_ip', 'popular_email',
       'email_equal', 'ip_equal', 'place_number', 'freq', 'hour',
       'fingerprint', 'is_eq']

ohe_cols = ['paymentSystem',
 'providerId',
 'bankCountry',
 'partyId',
 'shopId',
 'currency',
 'flag_ip',
 'flag_ip_party',
 'card_ip',
 'card_email',
 'nan_ip',
 'nan_email',
 'popular_ip',
 'popular_email',
 'email_equal',
 'ip_equal',
 'place_number',
 'freq',
 'hour']


float_cols = ['amount', 'cnt', 'sum', 'providerId']

def get_features(df):
    df['is_eq'] = (df['amount'] * df['sum'] != df['cnt']).astype(int)


    flag_ips = df.groupby('ip')['shopId'].agg(lambda x: len(set(x)))
    flag_ips_party = df.groupby('ip')['partyId'].agg(lambda x: len(set(x)))


    card_email = df.groupby('cardToken')['email'].agg(lambda x: str(len(set(x.dropna()))))
    card_ip = df.groupby('cardToken')['ip'].agg(lambda x: str(len(set(x.dropna()))))



    df['flag_ip'] = df['ip'].apply(lambda x: str(flag_ips[x]) if type(x) == str else '-1')
    df['flag_ip_party'] = df['ip'].apply(lambda x: str(flag_ips_party[x]) if type(x) == str else '-1')

    df['card_ip'] = df['cardToken'].apply(lambda x: card_ip[x])
    df['card_email'] = df['cardToken'].apply(lambda x: card_email[x])


    nan_ip = df.groupby('cardToken')['ip'].agg(lambda x: x.isna().sum()/ len(x))
    df['nan_ip'] = df['cardToken'].apply(lambda x: nan_ip[x] // 0.1).astype(str)

    nan_email = df.groupby('cardToken')['email'].agg(lambda x: x.isna().sum()/ len(x))
    df['nan_email'] = df['cardToken'].apply(lambda x: nan_email[x] // 0.1).astype(str)


    popular_ip = df['ip'].value_counts()[:150].apply(lambda x: str(int(math.log(x))))
    df['popular_ip'] = df['ip'].apply(lambda x: popular_ip[x] if x in popular_ip.index else -1)

    popular_email = df['email'].value_counts()[:150].apply(lambda x: str(int(math.log(x))))
    df['popular_email'] = df['email'].apply(lambda x: popular_email[x] if x in popular_email.index else -1)

    email_counts = df.groupby('email')['cardToken'].agg(lambda x: str(len(set(x.dropna()))))
    df['email_equal'] = df['email'].apply(lambda x: email_counts[x] if type(x) == str else '-1')

    ip_counts = df.groupby('ip')['cardToken'].agg(lambda x: str(len(set(x.dropna()))))
    df['ip_equal'] = df['ip'].apply(lambda x: ip_counts[x] if type(x) == str else '-1')

    place_number = df.groupby('cardToken')[['shopId', 'partyId']].agg(lambda x: len(set(x))).sum(axis=1)
    df['place_number'] = df['cardToken'].apply(lambda x: str(place_number[x]))




    df['eventTime'] = df['eventTime'].apply(lambda x: pd.Timestamp(x))
    freq = df.groupby('cardToken')['eventTime'].agg(lambda x: (max([i.hour for i in x]) - min([i.hour for i in x])) / len(x))
    df['freq'] = df['cardToken'].apply(lambda x: str(freq[x]//1))


    df['hour'] = df['eventTime'].apply(lambda x: str(x.hour))

    return df