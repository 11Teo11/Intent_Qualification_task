## Approach

My approach evolved through several iterations to find balance between accuracy, speed, and scalability. I designed a pipeline that uses an LLM for precise upfront filtering, followed by a hybrid ranking mechanism (Semantic + Lexical) with dynamic weighting. 

The architecture includes the following components and steps:

### 1. Data Formatting and Pre-processing

I noticed that fields like `address`, `primary_naics`, and `secondary_naics` were stored as strings, for some companies, containing "hidden dictionaries" or lists. To parse these faster, I created a custom parser using `ast.literal_eval` instead of `json.loads()`, since some of the strings contained single quotes (`'`) instead of standard double quotes (`"`).
* **Initial thought:** To keep the code completely modular, I wanted to dynamically test every column to see if it needed formatting.
* **Issue:** When testing this dynamic approach on a scaled dataset (simulating 100k elements), the processing time exploded.
* **Tradeoff/Solution:** I opted for a more scalable approach by hardcoding a specific list of columns that required formatting (`PARSE_COLS`).

---

### 2. LLM Extraction and Safe Filtering

To narrow down candidates efficiently, I used `Gemini 2.5-flash` to extract specific criteria from the user query to act as "hard filters".
* **Initial thought:** Extract a wide range of criteria (industry, business model, country code, NAICS, etc.).
* **Issue:** This caused the hard-filtering to be too aggressive, frequently resulting in an empty list of candidates before they could even reach the next stage of the ranking system.
* **Solution:** I reduced the LLM's scope extract only strict geographical data (country, region, town) and public status. *(Note: This was not the final solution and is addressed later).*

---

### 3. Semantic Ranking and Query Cleaning

For the ranking phase, I mapped the remaining companies using `Cosine Similarity`. The target document string was formed by concatenating `operational_name`, `description`, `core_offerings`, and `target_markets`. I encoded this and the user query using a local embedding model (`all-MiniLM-L6-v2`).
* **Test Case:** *"Internation trade and investment company in Constanța dealing with industrial supplies"* (Targeting a company named "Valbur").
* **Issue:** "Portul Constața" ranked 1st (52.66% matching) purely because the word "Constața" appeard in its name. Valbur ranked 2nd and there was another firm with a very close matching score, AQUA CENTER. The semantic model was heavily biased by the location name, which we had already used as a filter.
* **Solution:** I needed to remove the criteria already used for filtering from the query. However, the LLM translates values (e.g., extracting "ro" when the user types "Romania"), so a simple string replace wouldn't work. I updated the LLM prompt to explicitly return a list of the *exact original phrases* from the user's text so I could cleanly remove them.
* **Result:** After cleaning the query, "AQUA CENTER" took the 1st place, Valbur 2nd, and Portul Constanta dropped to 3rd. Valbur was close, but since the embedding model was small and local, it struggled to perfectly differentiate the complex industrial context. Cosine similarity wasn't enough.

---

### 4. Hybrid Ranking (Adding BM25)

To fix the context nuances, I realized I needed a way to reward exact keyword matches alongside semantic meaning.
* **Solution:** I introduced `BM25` (a lexical algorithm that scores exact word occurences) to complement `Cosine Similarity` (which understands context/synonyms). I normalized both scores and averaged them (50/50 split).
* **Result:** Valbur successfully jumped to 1st place (67.52%)

---

### 5. Dynamic Weighting

While the 50/50 split worked for the previous query, I tested a longer, more descriptive query.
* **Test Case:** *"Companies that supply packaging materials for direct-to-consumer cosmetic brands"*.
* **Issue:** A company named "FLextribe" took 1st place with an 75.17% total score (`sem: 0.50 | bm25: 1.00`). It  won purely because it had 100% BM25 score (an exact keyword match for "direct-to-consumer"), even though its semantic score (50%) showed it wasn't a great contextual fit compared to companies that primarly dealt with cosmetic packaging (which had semantic scores over 65%).
* **Solution:** A static 50/50 split allows BM25 to easily hijack long descriptive queries if it finds a keyword match. I implemented dynamic weight based on the length of the query (excluding the words removed by the LLM):
    * **<= 3 words:** 30% Semantic / 70% BM25 *(Favors exact matches for precise, short searches).*
    * **<= 7 words:** 50% Semantic / 50% BM25.
    *  **> 7 words:** 70% Semantic / 30% BM25 *(Favors context over keywords for long descriptions).*
* **Result:** With dynamic weights and normalized scores, the system correctly prioritized the context. Companies that deal with cosmetic packaging (primarly) dominated the top results:
```
87.58% Shanghai Bochen Cosmetic Packaging - cncosmeticbottles.com (sem: 0.96 | bm25: 0.68)
84.52% Shenzhen Itop Cosmetic Packaging - itoppacking.com (sem: 0.94 | bm25: 0.63)
84.32% Flextribe - flextribe.co (sem: 0.78 | bm25: 1.00)
83.79% SHEACK Packaging - sheackpkg.com (sem: 0.95 | bm25: 0.58)
```
*(Note: The scores weren't perfect, but there was an improvement. With more research on the word limit we could optimize this further).*   

---

### 6. The Number Problem and Safe Fallbacks

Even with hybrid ranking and dynamic weights, a query containing numerical constraints (e.g., *"more than 1000 employees"* or *"revenue over $50 million"*) failed.
* **Issue:** Neither Cosine Similarity nor BM25 handles mathematical logic well.
* **Initial thought:** I went back to the LLM criteria extraction step and added `min_employees` and `min_revenue` to the extraction list. But now the filtering function couldn't work since the initial criteria loop was based on column names from the JSON.
* **Modularity tradeoff:** Since these criteria required mathematical operations (`>=`), I had to break my strictly modular filtering loop and handle them as special cases. 
* **Issue:** Hard-filtering on these numbers was often too harsh because many companies have missing data. If a filter was applied blindly, the candidate list would frequently become empty.
* **Solution (`safe_filter`):** I implemented a *fallback mechanism*. After applying each extracted criterion, if the resulting candidate list becomes empty, the system discards that specific filter and rolls back to the previous non-empty list. Furthermore, the numerical filters were adjusted to be robust to missing data by accepting missing values, ensuring companies aren't wrongly excluded.

---

## Tradeoffs

* **Speed vs. Deep LLM Reasoning:** Sending every company to an LLM would yield high accuracy on complex queries, but it scales terribly. I traded that deep reasoning for a much faster Hybrid Ranking system, using the LLM only once per query for parameter extraction.
* **Modularity vs. Scalability:** I traded complete modularity (dynamically checking every column for dicts/lists) for hardcoded column parsing (`PARSE_COLS`) to drastically reduce processing time on large datasets.
* **Stop Word Noise vs. Simplicity:** By extracting and deleting exact phrases via the LLM's `words_to_remove`, the resulting query often contains leftover connecting words (noise). While this noise is ignored by BM25, it can slightly affect the length calculation for dynamic weighting. This was an acceptable tradeoff to avoid building a massive hardcoded list of stop words.
* **Hard Filtering vs. Data Retention:** Introducing numerical filters required hard comparisons (`>=`), which broke the standardized modularity of the filtering loop. However, the tradeoff was necessary because vector models cannot perform mathematical logic.


---

## Error Analysis

While the system handles a wide variety of queries, it still struggles in a few specific areas:

1. **Nuanced Business Relationships:** The system looks for semantic similarity and exact words. If a query asks for *"Companies competing with traditional banks"*, the system might struggle if the companies don't explicitly state "we compete with banks" in their descriptions.
2. **Keyword Dominance on Mid-Length Queries:** While dynamic weighting fixed keyword dominance on *long* queries (like the packaging example), a 5 or 6-word query is still weighted 50/50. If a company has a perfect BM25 score for those 6 words, it can still occasionally outrank a slightly better semantic fit.

---

## Scaling

If the system needed to handle 100,000 companies per query instead of 500, the current "on-the-fly" Pandas operations and real-time embedding generation would become a massive bottleneck. To scale this, I would change the following:

1. **Vector Database:** Instead of generating and comparing `document_embeddings` in RAM via `cosine_similarity`, I would pre-calculate the vectors for all 100k companies at startup and store them in a dedicated vector database, which can search through indexes in milliseconds.
2. **BM25 Caching:** Similar to the vectors, the lexical matrix for BM25 would be pre-calculated and cached when the server loads, avoiding the need to tokenize 100,000 texts on every single query.
3. **Batch Processing:** I would optimize parallel processing, moving the LLM from sequential mode to batch processing if there are multiple concurrent queries from different users.

---

## Failure Modes

The system might produce confident but incorrect results under the following conditions:

* **The Value Translation Trap:** If the LLM successfully extracts a filter (e.g., `min_employees: 1000`) but fails to accurately grab the exact `words_to_remove` (e.g., it misses the phrase "more than 1000 employees"), those words remain in the query. The semantic model will then try to find the concept of "1000 employees" in the text, distorting the rankings.
* **Production Monitoring:** To detect these failures in production, I would monitor the difference between the Semantic score and the BM25 score for the top-ranked result. If a company has a Semantic score of 1.0 and a BM25 score of 0.0, it is a strong indicator that the engine deviated too much from the requested keywords, relying entirely on a contextual assumption (or vice versa).

---

## Critical Thinking Summary

* **Where does the system work extremely well?** It excels at "mixed" queries that combine strict constraints (location, revenue) with broad conceptual needs (e.g., "fast-growing fintech"). The `safe_filter` ensures it never fails completely.
* **How robust is the system to missing data?** By utilizing the `candidates[column].isna()` logic, the system ensures that companies are not wrongly excluded simply because their financial data or employee counts are private or missing from the dataset. It gives them the "benefit of the doubt" and lets the semantic ranker decide their final fate. But a company that misses crucial data ( de completat )
* **What improvements would I prioritize next?** I would integrate a standard NLP library (like `nltk` or `spaCy`) to properly remove stop words (and, the, with) from the query *after* the LLM extraction. This would make the word-count for the Dynamic Weighting logic perfectly accurate.
