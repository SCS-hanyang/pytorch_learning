from nltk import word_tokenize
from nltk.corpus import stopwords
import re

example = ['A spectre is haunting Europe—the spectre of Communism.'
'All the powers of old Europe have entered into a holy alliance to exorcise this spectre: Pope and Tsar, '
'Metternich and Guizot, French Radicals and German police-spies.'
'Where is the party in opposition that has not been decried as Communistic by its opponents in power?'
'Where is the opposition that has not hurled back the branding reproach of Communism, against the more advanced opposition '
'parties, as well as against its reactionary adversaries?'
'Two things result from this fact:'
'I. Communism is already acknowledged by all European powers to be itself a power.'
'II. It is high time that Communists should openly, in the face of the whole world, publish their views, their aims, '
'their tendencies, and meet this nursery tale of the Spectre of Communism with a manifesto of the party itself.'
'To this end, Communists of various nationalities have assembled in London, and sketched the following Manifesto,'
'to be published in the English, French, German, Italian, Flemish, and Danish languages.'
'The history of all hitherto existing society is the history of class struggles.'
'Freeman and slave, patrician and plebeian, lord and serf, guild-master and journeyman, in a word, oppressor and oppressed, '
'stood in constant opposition to one another, carried on an uninterrupted, now hidden, now open fight, a fight that each time ended, either in a revolutionary reconstitution of society at large, or in the common ruin of the contending classes.'
'In the earlier epochs of history, we find almost everywhere a complicated arrangement of society into various orders, a manifold '
'gradation of social rank.'
'In ancient Rome we have patricians, knights, plebeians, slaves; in the Middle Ages, feudal lords, vassals, guild-masters,'
' journeymen, apprentices, serfs; in almost all of these classes, again, subordinate gradations.'
'The modern bourgeois society that has sprouted from the ruins of feudal society has not done away with class antagonisms.'
'It has but established new classes, new conditions of oppression, new forms of struggle in place of the old ones.'
'Our epoch, the epoch of the bourgeoisie, possesses, however, this distinctive feature: it has simplified the class antagonisms.'
'Society as a whole is more and more splitting up into two great hostile camps, into two great classes directly '
'facing each other—bourgeoisie and proletariat.'
'From the serfs of the Middle Ages sprang the chartered burghers of the earliest towns.'
'From these burgesses the first elements of the bourgeoisie were developed.'
'The discovery of America, the rounding of the Cape, opened up fresh ground for the rising bourgeoisie.'
'The East-Indian and Chinese markets, the colonization of America, trade with the colonies, the increase in the means of exchange and in commodities generally, gave to commerce, to navigation, to industry, an impulse never before known, and thereby, to the revolutionary element in the tottering feudal society, a rapid development.'
'The feudal system of industry, under which industrial production was monopolized by closed guilds, now no longer sufficed for the growing wants of the new markets.'
'The manufacturing system took its place.'
'The guild-masters were pushed on one side by the manufacturing middle class; division of labor between '
'the different corporate guilds vanished in the face of division of labor in each single workshop.'
'Meantime the markets kept ever growing, the demand ever rising.'
'Even manufacture no longer sufficed.'
'Thereupon, steam and machinery revolutionized industrial production.'
'The place of manufacture was taken by the giant, Modern Industry; the place of the industrial middle class by industrial millionaires, the leaders of whole industrial armies, the modern bourgeois.'
'Modern industry has established the world market, for which the discovery of America paved the way.'
'This market has given an immense development to commerce, to navigation, to communication by land.'
'This development has, in its time, reacted on the extension of industry; and in proportion as industry,'
' commerce, navigation, railways extended, in the same proportion the bourgeoisie developed, increased its capital, '
'and pushed into the background every class handed down from the Middle Ages.']

bag = []
word_to_index = {}

def build_bag_of_words(example):
    example[0] = re.sub('[^a-zA-Z ]', ' ', example[0])
    tokenized = word_tokenize(example[0])

    word_to_index = {}
    bag = []

    stopword = stopwords.words('english')

    tokenized = [word for word in tokenized if word not in stopword]

    for word in tokenized:
        if word not in word_to_index.keys():
            word_to_index[word] = len(word_to_index)
            bag.append(1)
        else:
            bag[word_to_index[word]] += 1

    return word_to_index, bag

vocab, bag = build_bag_of_words(example)
print(f"vocab \n {vocab}")
print(f'bag \b {bag}')