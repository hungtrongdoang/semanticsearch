import time
from elasticsearch import Elasticsearch
import streamlit as st
from sentence_transformers import SentenceTransformer
from pyvi.ViTokenizer import tokenize

@st.cache_resource
def load_es():
    # Tải mô hình Vietnamese-SBERT và phoBERT
    model_embedding_sbert = SentenceTransformer('saved_model_sbert')
    model_embedding_phobert = SentenceTransformer('saved_model')  # phoBERT (VoVanPhuc)
    client = Elasticsearch(hosts=["http://localhost:9200/"])
    return model_embedding_sbert, model_embedding_phobert, client

def embed_text(text, model_embedding):
    text_embedding = model_embedding.encode(text)
    return text_embedding.tolist()

def search(query, type_ranker, model_embedding, client):
    if type_ranker in ['SBERT', 'phoBERT']:
        time_embed = time.time()
        query_vector = embed_text([tokenize(query)], model_embedding)[0]
        print('TIME EMBEDDING ', time.time() - time_embed)
        script_query = {
            "script_score": {
                "query": {
                    "match_all": {}
                },
                "script": {
                    "source": "cosineSimilarity(params.query_vector, 'title_vector') + 1.0",
                    "params": {"query_vector": query_vector}
                }
            }
        }
    else:
        # BM25 search
        script_query = {
            "match": {
                "title": {
                    "query": query,
                    "fuzziness": "AUTO"
                }
            }
        }

    index_name = 'demo_simcse_v4' if type_ranker == 'SBERT' else 'demo_simcse_v3' if type_ranker == 'phoBERT' else 'demo_simcse_v3'

    response = client.search(
        index=index_name,
        body={
            "size": 100,
            "query": script_query,
            "_source": {
                "includes": ["id", "title"]
            },
        },
        ignore=[400]
    )

    result = []
    if 'hits' in response and 'hits' in response['hits']:
        for hit in response["hits"]["hits"]:
            result.append(hit["_source"]['title'])
    return result

def run():
    st.title('Vietnamese-SBERT & SimCSE Semantic Search')
    st.markdown('Here is an example')
    st.text('')

    comment = st.text_input('Write your test content!')

    if st.button('Search'):
        if comment.strip() == '':
            st.warning("Please enter some text for searching!")
        else:
            with st.spinner('Searching ......'):
                # Tìm kiếm với cả 3 phương pháp
                bm25_results = search(comment, 'BM25', None, client)
                sbert_results = search(comment, 'SBERT', model_embedding_sbert, client)
                phobert_results = search(comment, 'phoBERT', model_embedding_phobert, client)

                # Hiển thị kết quả theo 3 cột
                col1, col2, col3 = st.columns(3)

                with col1:
                    st.header("Results from BM25")
                    if bm25_results:
                        for i, res in enumerate(bm25_results):
                            st.success(f"{i+1}. {res}")
                    else:
                        st.info("No results from BM25")

                with col2:
                    st.header("Results from SBERT")
                    if sbert_results:
                        for i, res in enumerate(sbert_results):
                            st.success(f"{i+1}. {res}")
                    else:
                        st.info("No results from SBERT")

                with col3:
                    st.header("Results from phoBERT")
                    if phobert_results:
                        for i, res in enumerate(phobert_results):
                            st.success(f"{i+1}. {res}")
                    else:
                        st.info("No results from phoBERT")

if __name__ == '__main__':
    model_embedding_sbert, model_embedding_phobert, client = load_es()
    run()
