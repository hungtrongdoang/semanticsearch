import time
from elasticsearch import Elasticsearch
import streamlit as st
from sentence_transformers import SentenceTransformer
from pyvi.ViTokenizer import tokenize

# Load Elasticsearch client and embedding model
@st.cache_resource
def load_es():
    model_embedding = SentenceTransformer('saved_model')
    client = Elasticsearch(hosts=["http://localhost:9200/"])
    return model_embedding, client

# Function to embed text using the loaded model
def embed_text(text, model_embedding):
    text_embedding = model_embedding.encode(text)
    return text_embedding.tolist()

# Function to perform search using either BM25 or SimCSE
def search(query, type_ranker, model_embedding, client):
    if type_ranker == 'SimCSE':
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
    else:  # Use BM25 ranking
        script_query = {
            "match": {
              "title": {
                "query": query,
                "fuzziness": "AUTO"
              }
            }
        }

    # Perform the search on Elasticsearch
    response = client.search(
        index='demo_simcse_v2',  # Make sure to use the correct index name
        body={
            "size": 100,
            "query": script_query,
            "_source": {
                "includes": ["id", "title"]
            },
        },
        ignore=[400]
    )

    # Parse and return search results
    result = []
    if 'hits' in response and 'hits' in response['hits']:
        for hit in response["hits"]["hits"]:
            result.append(hit["_source"]['title'])
    return result

# Streamlit app UI and logic
def run():
    st.title('Test Semantic Search')
    ranker = st.sidebar.radio('Rank by', ["BM25", "SimCSE"], index=0)
    st.markdown('Here is an example')
    st.text('')

    # User input for query
    comment = st.text_input('Write your test content!')
    
    # Perform search when button is clicked
    if st.button('SEARCH'):
        if comment.strip() == '':
            st.warning("Please enter some text for searching!")
        else:
            with st.spinner('Searching ......'):
                if ranker == 'SimCSE':
                    result_ = search(comment, 'SimCSE', model_embedding, client)
                else:
                    result_ = search(comment, 'BM25', model_embedding, client)
                
                # Display search results
                if result_:
                    for i, res in enumerate(result_):
                        st.success(f"{i+1}. {res}")
                else:
                    st.info("No results found.")

if __name__ == '__main__':
    # Load the model and Elasticsearch client
    model_embedding, client = load_es()
    # Run the Streamlit app
    run()
