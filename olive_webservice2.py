import streamlit as st
from PIL import Image
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer, util
import torch
from konlpy.tag import Kkma
from transformers import AutoTokenizer, BartForConditionalGeneration

# 데이터 로드
data = pd.read_csv('C:/Users/11win/Desktop/olive_final/web/train_df.csv')

# Konlpy의 Kkma 사용
t = Kkma()
contents_tokens = [t.morphs(row) for row in data['bsisFacts']]
contents_for_vectorize = []
for content in contents_tokens:
    sentence = ''
    for word in content:
        sentence = sentence + ' ' + word
    contents_for_vectorize.append(sentence)

# TF-IDF 특징 벡터 추출
tfidf_vectorizer = TfidfVectorizer()
tfidf_matrix = tfidf_vectorizer.fit_transform(contents_for_vectorize).toarray()

# SBERT 특징 벡터 추출
sbert_model = SentenceTransformer('bert-base-multilingual-cased')
sbert_matrix = sbert_model.encode(data['bsisFacts'].tolist())

# TF-IDF 차원 축소
pca = PCA(n_components=sbert_matrix.shape[1])
tfidf_matrix_reduced = pca.fit_transform(tfidf_matrix)

# 이미지 로드
logo_img = Image.open('C:/Users/11win/Desktop/olive_final/web/logo.jpg')
st.sidebar.image(logo_img)
st.sidebar.title('여러분의 상황을 입력하세요. 가장 비슷한 판례를 찾아서 판결문을 요약해드립니다.')

logo_img2 = Image.open('C:/Users/11win/Desktop/olive_final/web/logo2.png')
st.subheader("법률 판례 검색 엔진")
st.image(logo_img2)

# 사용자 입력 받기
new_post = [st.text_area('당신의 사건을 알려주세요. (최대 200자)', max_chars=200)]

def find_similar_cases():
    new_post_tokens = [t.morphs(row) for row in new_post]
    new_post_for_vectorize = [' '.join(content) for content in new_post_tokens]

    new_post_str = ' '.join(new_post_for_vectorize)

    new_post_tfidf = tfidf_vectorizer.transform([new_post_str]).toarray()
    new_post_tfidf_reduced = pca.transform(new_post_tfidf)

    new_post_embedding = sbert_model.encode(new_post, convert_to_tensor=True)
    new_post_embedding_np = new_post_embedding.cpu().numpy()

    sbert_cosine_scores = util.pytorch_cos_sim(torch.tensor(new_post_embedding_np), torch.tensor(sbert_matrix))[0]
    tfidf_reduced_cosine_scores = cosine_similarity(new_post_tfidf_reduced, tfidf_matrix_reduced)

    best_alpha = 0.5
    best_beta = 0.5

    combined_cosine_scores = best_alpha * sbert_cosine_scores.cpu().numpy() + best_beta * tfidf_reduced_cosine_scores.flatten()

    top_results = np.argsort(-combined_cosine_scores)[:5]

    rows = []
    for idx in top_results:
        rows.append({'score': "{:.4f}".format(combined_cosine_scores[idx]),
                     'caseNo': data['caseNo'].iloc[idx],
                     'caseNm': data['caseNm'].iloc[idx],
                     'disposal': data['disposal'].iloc[idx],
                     'relateLaword': data['relateLaword'].iloc[idx],
                     'bsisFacts': data['bsisFacts'].iloc[idx],
                     'courtDcss': data['courtDcss'].iloc[idx],
                     'cnclsns': data['cnclsns'].iloc[idx]})

    df = pd.DataFrame(rows, columns=['score','caseNo','caseNm','disposal','relateLaword','bsisFacts','courtDcss','cnclsns'])

    # 모델 및 토크나이저 로드
    model_path = "C:/Users/11win/Desktop/olive_final/web/oliveKobart12"
    model = BartForConditionalGeneration.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained("ainize/kobart-news")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # 요약 생성
    summaries = []
    for row in df['courtDcss']:
        input_ids = tokenizer.encode(row, return_tensors="pt", max_length=1024, truncation=True).to(device)
        generated_ids = model.generate(input_ids, max_length=150, num_beams=4, length_penalty=2.0, early_stopping=True)
        generated_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
        summaries.append(generated_text)

    df['summary'] = summaries

    # 결과 출력
    st.write("당신의 사건과 유사한 판례 TOP5")
    for i in range(len(df)):
        st.write(f"{i + 1}번째로 유사한 사건")
        st.write(f"유사도:\n{float(df['score'].iloc[i]) * 100:.2f}%")
        st.write("사건 번호:\n", df['caseNo'].iloc[i])
        st.write("사건명:\n", df['caseNm'].iloc[i])
        st.write("기초사실:\n", df['bsisFacts'].iloc[i])
        st.write("재판부의 판단(요약 제공):\n", df['summary'].iloc[i])
        st.write("================================================================================")


if st.button('유사도 판례 찾기'):
    find_similar_cases()
