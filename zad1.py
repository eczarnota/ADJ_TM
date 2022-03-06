import re

input = "Dzisiaj mamy 4 stopnie na plusie, 1 marca 2022 roku"
result = re.sub('\d', '', input)
print(result)

input = "<div><h2>Header</h2> <p>article<b>strong text</b> <a href="">link</a></p></div>"
result = re.sub('<[^>]*>', '', input)
print(result)

input = "Lorem ipsum dolor sit amet, consectetur; adipiscing elit.Sed eget mattis sem. Mauris egestas erat quam, ut faucibus eros congue et. Inblandit, mi eu porta; lobortis, tortor nisl facilisis leo, at tristique augue risuseu risus."
result = re.sub('[!"#$%&\'()*+,\-./:;<=>?@[\]^_`{|}~]', '', input)
print(result)
