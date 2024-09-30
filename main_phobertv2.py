import time
from elasticsearch import Elasticsearch
import streamlit as st
from sentence_transformers import SentenceTransformer
from transformers import AutoModel, AutoTokenizer
import torch
from pyvi.ViTokenizer import tokenize

@st.cache_resource
def load_es():
    # Load Vietnamese-SBERT and SimCSE PhoBERT (VoVanPhuc)
    model_embedding_sbert = SentenceTransformer('saved_model_sbert')
    model_embedding_phobert = SentenceTransformer('saved_model')  # SimCSE PhoBERT (VoVanPhuc)

    # Load PhoBERT v2
    model_embedding_phobert_v2 = AutoModel.from_pretrained('phobert-base-v2-model')
    tokenizer_phobert_v2 = AutoTokenizer.from_pretrained('phobert-base-v2-model')

    # Elasticsearch client
    client = Elasticsearch(hosts=["http://localhost:9200/"])
    return model_embedding_sbert, model_embedding_phobert, model_embedding_phobert_v2, tokenizer_phobert_v2, client

def embed_text(text, model_embedding):
    """Embed text for SBERT and SimCSE PhoBERT"""
    text_embedding = model_embedding.encode(text)
    return text_embedding.tolist()

def embed_phobert_v2(text, model_embedding, tokenizer):
    """Embed text for PhoBERT v2"""
    inputs = tokenizer(text, padding=True, truncation=True, return_tensors="pt")
    with torch.no_grad():
        outputs = model_embedding(**inputs)
    return outputs.last_hidden_state.mean(dim=1).tolist()

def search(query, type_ranker, model_embedding=None, tokenizer=None, client=None):
    """Perform search across different rankers"""
    if type_ranker in ['SBERT', 'phoBERT']:
        query_vector = embed_text([tokenize(query)], model_embedding)[0]
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
    elif type_ranker == 'phoBERT_v2':
        query_vector = embed_phobert_v2(query, model_embedding, tokenizer)[0]
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

    index_name = (
        'demo_simcse_v4' if type_ranker == 'SBERT' 
        else 'demo_simcse_v3' if type_ranker == 'phoBERT' 
        else 'demo_simcse_phobertv2' if type_ranker == 'phoBERT_v2' 
        else 'demo_simcse_v3'
    )

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

    comment = st.text_input('Write your test content!')

    if st.button('Search'):
        if comment.strip() == '':
            st.warning("Please enter some text for searching!")
        else:
            with st.spinner('Searching ......'):
                # Perform searches with all methods
                bm25_results = search(comment, 'BM25', client=client)
                sbert_results = search(comment, 'SBERT', model_embedding_sbert, client=client)
                phobert_results = search(comment, 'phoBERT', model_embedding_phobert, client=client)
                phobert_v2_results = search(comment, 'phoBERT_v2', model_embedding_phobert_v2, tokenizer_phobert_v2, client)

                # Display results in 4 columns
                col1, col2, col3, col4 = st.columns(4)

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
                    st.header("Results from SimCSE PhoBERT")
                    if phobert_results:
                        for i, res in enumerate(phobert_results):
                            st.success(f"{i+1}. {res}")
                    else:
                        st.info("No results from SimCSE PhoBERT")

                with col4:
                    st.header("Results from PhoBERT v2")
                    if phobert_v2_results:
                        for i, res in enumerate(phobert_v2_results):
                            st.success(f"{i+1}. {res}")
                    else:
                        st.info("No results from PhoBERT v2")

if __name__ == '__main__':
    model_embedding_sbert, model_embedding_phobert, model_embedding_phobert_v2, tokenizer_phobert_v2, client = load_es()
    run()
