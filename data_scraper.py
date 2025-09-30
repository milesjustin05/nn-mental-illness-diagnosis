# Thanks to youtube tutorial by James Briggs https://www.youtube.com/watch?v=FdjVoOf9HN4
# I have no idea what most of this does, but the code works so hey, can not complain.

import requests

# API public key
CLIENT_ID = 'PS6DK-y7m18lVz3h4KYwmA'

# Secret key which is in a file on my computer not posted into github
with open('sk.txt', 'r') as f:
    SECRET_KEY = f.read()

auth = requests.auth.HTTPBasicAuth(CLIENT_ID, SECRET_KEY)

# Reddit account password, also hidden on a non posted file on my personal pc
with open('pw.txt', 'r') as f:
    pw = f.read()

# Reddit account login stuffs
data = {
    'grant_type': 'password',
    'username': 'Hot-Border-2234',
    'password': pw
}

# I have no clue what this does
headers = {'User-Agent': 'MyApi/0.0.1'}

# I believe this verifies the code with the api using my reddit login
res = requests.post('https://www.reddit.com/api/v1/access_token',
                    auth=auth, data=data, headers=headers)

#No clue what these do either
TOKEN = res.json()['access_token']
headers['Authorization'] = f'bearer {TOKEN}'

#grabs all the "hpt" posts from the ADHD subreddit
res = requests.get('https://oauth.reddit.com/r/ADHD/hot', headers=headers)

#prints the hot posts in to a hot clutter of information
print(res.json())