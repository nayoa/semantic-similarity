import spacy

nlp = spacy.load('en_core_web_md')
nlp_sm = spacy.load('en_core_web_sm')

## web_md language model
word1 = nlp("cat")
word2 = nlp("monkey")
word3 = nlp("banana")
word4 = nlp("capybara")
word5 = nlp("hamster")
word6 = nlp("dog")

print("web_md language model")
print(word1.similarity(word2))
print(word3.similarity(word2)) # Are these similar because monkeys "commonly" are holding a banana in pictures?
print(word3.similarity(word1))
print(word4.similarity(word1))
print(word5.similarity(word2))
print(word6.similarity(word5))

## web_sm language model
word7 = nlp_sm("cat")
word8 = nlp_sm("monkey")
word9 = nlp_sm("banana")
word10 = nlp_sm("capybara")

## Using "web_sm" language model has no word vectors loaded
## The result is based on the tagger, parser and NER, which may not give useful similarity judgements
## The result alo gives a warning message -> [W007]
print("web_sm language model")
print(word7.similarity(word8))
print(word9.similarity(word8))
print(word9.similarity(word7))
