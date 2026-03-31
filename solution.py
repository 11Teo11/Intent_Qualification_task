import pandas as pd
import ast
import json
import numpy as np
import os
from google import genai
from google.genai import types
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from rank_bm25 import BM25Okapi

API_KEY = os.getenv("GEMINI_API_KEY")

if not API_KEY:
    print("Error: Please set the GEMINI_API_KEY")
    exit()

PARSE_COLS = ["address", "primary_naics", "secondary_naics"]

client = genai.Client(api_key = API_KEY)
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')


def load_and_format_data(filepath):
    df = pd.read_json(filepath, lines=True)

    def parse_string(x):
        if isinstance(x,str):
            x = x.strip()
            if len(x) > 0 and (x[0] == '{' and x[-1] == '}') or (x[0] == '[' and x[-1] == ']'):
                try:
                    return ast.literal_eval(x)
                except (ValueError, SyntaxError):
                    return x
        return x

    for col in PARSE_COLS:
        if col in df.columns:
            df[col] = df[col].apply(parse_string)
        
    return df

def extract_query_criteria(text):
    prompt = """
    You are a data extraction assistant. Analyze the user's search query for companies and extract the filtering criteria into a valid JSON format.
    Possible keys to extract:
    - "country_code" (2-letter lowercase code, e.g., "ro" for Romania, "de" for Germany)
    - "region" (string, e.g., Georgia)
    - "town" (string, e.g., Bucharest)
    - "is_public" (boolean: true or false)
    - "min_employees" (integer, extract the minimum number of employees requested)
    - "min_revenue" (integer, extract the minimum revenue requested is USD)
    - "words_to_remove" (list of strings: extract the EXACT original phrases from the user's query that correspond to ANY of the extracted criteria above. This includes the country names, and the full context for numbers, e.g. ["Romania", "Publicly traded", "with more than 1000 employees", "revenue over 1 million USD"])
    
    If a criterion is not mentioned, do not include its key in the JSON.
    Return ONLY the valid JSON, without markdown formatting or other text.

    User query: 
    """

    try:
        config = types.GenerateContentConfig(response_mime_type="application/json", temperature=0.0)

        query_text = f"{prompt} {text}"
        
        response = client.models.generate_content(
            model = 'gemini-2.5-flash',
            contents = query_text,
            config = config
        )
        
        return json.loads(response.text)
    
    except Exception as e:
        print(f"Error LLM call: {e}")
        return {}

def apply_filters(df, criteria):
    candidates = df.copy()

    def safe_filter(current_df, mask):
        filtered_df = current_df[mask]
        if filtered_df.empty:
            return current_df
        return filtered_df

    if "min_employees" in criteria:
        mask = (candidates["employee_count"] >= criteria["min_employees"]) | candidates["employee_count"].isna()
        candidates = safe_filter(candidates, mask)

    if "min_revenue" in criteria:
        mask = (candidates["revenue"] >= criteria["min_revenue"]) | candidates["revenue"].isna()
        candidates = safe_filter(candidates, mask)


    for key,val in criteria.items():

        if key in ["words_to_remove","min_employees","min_revenue"]:
            continue

        if key in candidates.columns:
            def check_list(cell, target_val=val):
                if isinstance(cell, list):
                    return str(target_val).lower() in [str(i).lower() for i in cell]
                return str(cell).lower() == str(target_val).lower()
            
            mask = candidates[key].apply(check_list)
            candidates = safe_filter(candidates, mask)
        else:
            mask = pd.Series(False, index = candidates.index)
            for col in PARSE_COLS:
                def check_dict(cell, k = key, v = val):
                    if isinstance(cell,dict) and k in cell:
                        return str(cell[k]).lower() == str(v).lower()
                    return False
                mask = mask | candidates[col].apply(check_dict)
            candidates = safe_filter(candidates, mask)

    return candidates

def clean_query(query, criteria):
    query = query.lower()

    if "words_to_remove" in criteria:
        for phrase in criteria["words_to_remove"]:
            query = query.replace(phrase.lower(), "")

    return " ".join(query.split())

def calculate_weights(query):
    word_cnt = len(query.split())

    if word_cnt <= 3:
        semantic_weight = 0.3
        bm25_weight = 0.7

    elif word_cnt <= 7:
        semantic_weight = 0.5
        bm25_weight = 0.5

    else:
        semantic_weight = 0.7
        bm25_weight = 0.3

    return semantic_weight, bm25_weight

def rank_candidates(query, candidates_df, top_k=10):

    if candidates_df.empty:
        return candidates_df
        
    ranked_df = candidates_df.copy()
    
    def clean_list(item):
        if isinstance(item, list):
            return " ".join([str(i) for i in item])
        return str(item) if pd.notna(item) else ""

    ranked_df['search_text'] = (
        ranked_df['operational_name'].fillna("") + ". " + 
        ranked_df['description'].fillna("") + " " +
        ranked_df['core_offerings'].apply(clean_list) + " " +
        ranked_df['target_markets'].apply(clean_list)
    )
    
    document_embeddings = embedding_model.encode(ranked_df['search_text'].tolist())
    query_embedding = embedding_model.encode([query])
    
    sem_scores = cosine_similarity(query_embedding, document_embeddings)[0]
    
    words_list = [str(i).lower().split() for i in ranked_df['search_text']]
    bm25 = BM25Okapi(words_list)
    bm25_scores = np.array(bm25.get_scores(query.lower().split()))

    def normalize(scores):
        s_min, s_max = scores.min(), scores.max()
        if s_max > s_min:
            return (scores - s_min) / (s_max - s_min)
        else:
            return np.zeros(len(scores))
 

    sem_scaled = normalize(sem_scores)
    bm25_scaled = normalize(bm25_scores)
    
    sem_weight, bm25_weight = calculate_weights(query)

    ranked_df['semantic_score'] = sem_scaled
    ranked_df['bm25_score'] = bm25_scaled
    ranked_df['final_score'] = (sem_scaled * sem_weight) + (bm25_scaled * bm25_weight)
    
    ranked_df = ranked_df.sort_values(by='final_score', ascending=False)
    
    return ranked_df.head(top_k)


if __name__ == "__main__":
    companies_df = load_and_format_data("companies.jsonl")
    
    while True:
        user_query = input("Enter your query (or type 'exit' to quit): ").strip()

        if user_query.lower() == "exit":
            break

        criteria = extract_query_criteria(user_query)
        print(f"Filters detected: {criteria}\n")

        candidates = apply_filters(companies_df, criteria)

        semantic_query = clean_query(user_query, criteria)

        top_results = rank_candidates(semantic_query, candidates, top_k = 10)

        if top_results.empty:
            print("No companies found")
        else:
            print("Top results:\n")
            for _, row in top_results.iterrows():
                print(f"{(row['final_score']*100):.2f}% {row['operational_name']} - {row['website']} (sem: {row['semantic_score']:.2f} | bm25: {row['bm25_score']:.2f})")

        print("-" * 30 + "\n")
