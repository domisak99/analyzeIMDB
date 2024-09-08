import requests
from bs4 import BeautifulSoup
import nltk
from nltk.tokenize import RegexpTokenizer
from nltk.probability import FreqDist
from nltk.sentiment import SentimentIntensityAnalyzer
from nltk.corpus import stopwords
from wordcloud import WordCloud, STOPWORDS
import matplotlib.pyplot as plt

url1 = 'https://www.imdb.com/title/tt0111161/reviews?ref_=tt_sa_3'
url2 = 'https://www.imdb.com/title/tt0329774/reviews?ref_=tt_ql_3'
url3 = 'https://www.imdb.com/title/tt0401855/reviews/?ref_=tt_ql_urv'

stopwords = set(stopwords.words('english'))

def get_wordcloud(sentiment, countwords, longestwords):
    wordcloudOfSentiment = WordCloud(
        background_color='white',
        stopwords=stopwords,
        max_font_size=200,
        scale=3
    ).generate(str(sentiment))
    
    wordcloudOfCountWords = WordCloud(
        background_color='white',
        stopwords=stopwords,
        max_font_size=200,
        scale=3
    ).generate(str(countwords))

    wordcloudOfLongestWords = WordCloud(
        background_color='white',
        stopwords=stopwords,
        max_font_size=200,
        scale=3
    ).generate(str(longestwords))

    wc = plt.figure(1, figsize=(10, 10))
    wc = plt.gcf()
    wc.canvas.manager.set_window_title('Wordcloud zobrazení')
    
    wc.add_subplot(2, 2, 1)
    plt.axis('off')
    plt.imshow(wordcloudOfSentiment)
    plt.title("Sentiment")

    wc.add_subplot(2, 2, 2)
    plt.axis('off')
    plt.imshow(wordcloudOfCountWords)
    plt.title("30 most used words")

    wc.add_subplot(2, 2, 3)
    plt.axis('off')
    plt.imshow(wordcloudOfLongestWords)
    plt.title("30 most longest words")

    plt.show()

def analyze_website(url):
    reviews_text = []
    score_board = [0,0,0]
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')

    movie_name_str = soup.findAll('a', attrs={"itemprop":"url"})
    for name in movie_name_str:
        movie_name=name.text

    reviews_text_str = soup.findAll('div', attrs={"class":"text show-more__control"})
    for review in reviews_text_str:
        reviews_text.append(review.text)

    #pozitivni 0, neutralni 1, negativni 2
    states=['Positive','Negative','Neutral']

    for review in reviews_text:
        sia = SentimentIntensityAnalyzer()
        score = sia.polarity_scores(review)
        if score['compound'] >= 0.05 :
            score_board[0]+=1
     
        elif score['compound'] <= - 0.05 :
            score_board[1]+=1
     
        else :
            score_board[2]+=1

    tokenizer = RegexpTokenizer(r'\w+')
    wordsList = tokenizer.tokenize(str(reviews_text).lower());
    wordsListAfterStopwords = [i for i in wordsList if i not in stopwords]


    fdist = FreqDist(word.lower() for word in wordsListAfterStopwords)
    wordCountList = fdist.most_common(30)

    WordListByLongest = sorted(wordsListAfterStopwords, key=len, reverse=True)
    First30LongestWords = WordListByLongest[0:30]
    
    print("URL = ", url)
    print("Reviews = ", len(reviews_text))
    print("Name = ", movie_name)

    print("\nSentiment")
    for i in range(3):
        print(states[i] + " - " + str(score_board[i]))

    print("\n30 most used words:")
    for item in wordCountList:
        printStr = item[0] + " -> " + str(item[1]) + ", ";
        print(printStr, end = '')



    print("\n\n30 most longest words:")
    for item in First30LongestWords:
        printStr = item + ", "
        print(printStr, end = '')

    sentiment_array = []
    for i in range(3):
        sentiment_array.append([states[i],score_board[i]])

    get_wordcloud(sentiment_array, wordCountList, First30LongestWords);

analyze_website(url1)
print("\n")
analyze_website(url2)
print("\n")
analyze_website(url3)

