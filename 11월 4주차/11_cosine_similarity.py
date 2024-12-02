import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

data = pd.read_csv('data/movies_metadata.csv', low_memory=False)

data = data.head(20000)

# data['overview'].isnull().sum : overview열에 null값을 모두 더해줌

data.loc[:,'overview'] = data['overview'].fillna('')          # null 값을 ''로 대체


tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(data['overview'])

cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

title_to_index = dict(zip(data['title'], data.index))

def get_recommendations(title, cosine_sim=cosine_sim):
    idx = title_to_index[title]

    sim_scores = list(enumerate(cosine_sim[idx]))

    sim_scores = sorted(sim_scores, key=lambda x:x[1], reverse=True)

    recommend_idx = [idx for idx, value in sim_scores[1:10]]

    return data['title'].iloc[recommend_idx]                    # pd.iloc[row_selection, column_selection]
                                                                # row_selection과 column_selection에는 iterable한 객체 사용 가능
                                                                # df.iloc[df['Age'] >= 30].iloc[:, 0]처럼 조건부로도 사용 가능
print(get_recommendations('The Dark Knight Rises'))
