from bs4 import BeautifulSoup
import requests, urllib2, os, sys, time, random

BASE_URL = 'http://www.azlyrics.com/'

def download_songs(url):
    time.sleep(random.random()*0.5)
    try:
        opener = urllib2.build_opener()
        opener.addheaders = [('User-agent', 'Mozilla/5.0')]
        page = urllib2.urlopen(url, proxies=proxies[0]).read()
        soup = BeautifulSoup(page, 'html.parser')

        # Get the artist name
        artist_name = soup.findAll('h1')[0].get_text()[:-7].lower().replace(' ', '_')

        # Store all songs for a given artist
        with open('data/input/'+artist_name+'.txt', 'wb') as w:
            for song in soup.findAll('a', {'target': '_blank'}):
                if 'lyrics/' in song['href']:
                    song_url = song['href'][1:].strip()
                    w.write(song_url + '\n')
    except urllib2.HTTPError:
        print '404 not found'

def download_lyrics(artist, url):
    print url
    time.sleep(random.random()*5)
    urllib2.install_opener(urllib2.build_opener(urllib2.ProxyHandler({'http': '66.180.180.117'})))
    page = urllib2.urlopen(url).read()
    soup = BeautifulSoup(page, 'html.parser')

    # Get the song title
    song_title = soup.find('title').get_text().split(' - ')[1].lower().replace('/', ' ').replace(' ', '_')

    # Get the lyrics div
    lyrics = soup.findAll('div', {'class': ''})

    for i in lyrics:
        lyrics = i.get_text().strip()
        if len(lyrics) > 10:
            with open('data/input/' + artist + '/' + song_title + '.txt', 'wb') as w:
                cleaned_lyrics = lyrics.replace('\r\n', '\n').replace('  ', ' ')
                w.write(cleaned_lyrics.encode('utf-8'))

def download_all_lyrics(artist):
    if not os.path.exists('data/input/'+artist):
        os.mkdir('data/input/'+artist)

    with open('data/input/'+artist+'.txt', 'r') as songs:
        for song in songs.readlines():
            url = BASE_URL + song[2:].strip()
            download_lyrics(artist, url)

def get_article_text(url):
    article_text = ""
    r = requests.get(url)
    soup = BeautifulSoup(r.text, "lxml")
    for this_div in soup.find_all("p", class_="story-body-text story-content"):
        article_text += this_div.get_text() + '\n\n'

    return article_text

# url = "http://www.azlyrics.com/e/eminem.html"
# download_songs(url)
# download_all_lyrics("eminem")

# url = "https://www.nytimes.com/2017/05/11/opinion/once-syrian-kurds-beat-isis-dont-abandon-them-to-turkey.html"
# url = "https://www.nytimes.com/2017/05/11/world/europe/pineapple-art-scotland.html"
url = "https://www.nytimes.com/2017/05/11/us/politics/for-trump-supporters-the-real-outrage-is-the-lefts-uproar-over-comey.html"
print get_article_text(url).encode("ascii", "ignore")
