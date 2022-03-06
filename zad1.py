import re
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords

stops = set(stopwords.words('english'))
print(stops)
def OczyszczanieTekstu(input):
    result = re.findall('[;:][^\w|\s|;|:]?[^\w|\s|;|:]', input)
    result1 = re.sub('[;:][^\w|\s|;|:]?[^\w|\s|;|:]', '', input)
    result2 = result1.lower()
    result3 = re.sub('\d', '', result2)
    result4 = re.sub('<[^>]*>', '', result3)
    result5 = re.sub('[!"#$%&\'()*+,\-./:;<=>?@[\]^_`{|}~]', '', result4)
    result6 = " ".join(result5.split())
    result7=result6
    for emotikony in result:
        result7 += emotikony
    print(result7)
    return result7

OczyszczanieTekstu("<div><h2>  :)  Header333,,  </h2> <p>article1<b>strong   ;)   text2!</b> <a href="">link     </a>:( </p></div>")
