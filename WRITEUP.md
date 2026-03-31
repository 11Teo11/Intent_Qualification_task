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

While the system handles a wide variety of queries, it still struggles in a few specific areas involving complex relationships and geographical generalizations.

* **Test Case:** *"Fast-growing fintech companies competing with traditional banks in Europe."*
* **Extracted Filters:** `{'region': 'Europe', 'words_to_remove': ['Europe']}`
* **Top Results:**
```
77.05% European Pay - european-pay.fr (sem: 0.67 | bm25: 1.00)       
73.21% New Payment Innovation - npi.ie (sem: 1.00 | bm25: 0.11)      
68.81% Auvergnat Cola - auvergnatcola.com (sem: 0.58 | bm25: 0.93)   
68.38% FGC Money Transfer - fgcexchange.co.uk (sem: 0.64 | bm25: 0.79)
67.19% CTC FUND - ctcfundlimited.com (sem: 0.83 | bm25: 0.31)        
67.16% Telefónica Venture Builder - telefonica.com (sem: 0.71 | bm25: 0.58)
66.26% Euro-Rijn Financial Services - erfinancial.services (sem: 0.78 | bm25: 0.38)
65.54% Rantum Capital - rantumcapital.de (sem: 0.80 | bm25: 0.33)    
65.24% Tietoevry - tietoevry.com (sem: 0.60 | bm25: 0.78)
62.31% Ferring - ferring.com (sem: 0.75 | bm25: 0.32)
```
* **Why it failed:**
1. **The Geographical Generalization Trap:** The LLM extracted `"Europe"` as a `region`. However, the dataset's `region_name` generally contains specific states/counties, not continents. Because no company matched "Europe" exactly, the `safe_filter` triggered, effectively canceling the location filter and doing a global search.
2. **Stop-Words Noise (The Auvergnat Cola Error):** After removing the word "Europe", the query still contained words like *"with"*, *"in"*, and *"companies"*. Because I didn't implement a strict NLP stop-word remover, BM25 matched these highly common prepositions against random companies. This is why *Auvergnat Cola* (a French regional soda company) bizarrely reached 3rd place with a massive BM25 score of 0.93.
3. **The Relational Trap:** The system retrieved companies like *Tietoevry* (Enterprise IT). The semantic model and BM25 recognized the correct industry domain (they saw the words "banks" and "fintech"), but neither algorithm can understand the relational constraint *"competing with"*. It retrieved software vendors that *serve* banks, rather than startups that *compete* with them.

---

## Scaling

If the system needed to handle 100,000 companies per query instead of 500, the current Pandas operations and real-time embedding generation would become a massive bottleneck. To scale this, I would change the following:

1. **Better Models:** I used a small, local embedding model (`all-MiniLM-L6-v2`) and a fast LLM (`Gemini 2.5-flash`). For a large-scale app, I would upgrade to a more advanced embedding model that understands complex contexts much better, and a stronger LLM to avoid extraction mistakes (like confusing a continent with a region).
2. **Vector Database:** Instead of calculating `cosine_similarity` in memory every time, I would generate the embeddings for all 100k companies once at startup and save them in a Vector Database. This makes searching through massive datasets almost instant.
3. **BM25 Caching:** Just like the embeddings, breaking down 100,000 company descriptions into words for BM25 takes too much time if done on every query. I would pre-calculate and save this data too.

---

## Failure Modes

The system might produce confident but incorrect results under the following conditions:

* **The Value Translation Trap:** If the LLM successfully extracts a filter (e.g., `min_employees: 1000`) but fails to accurately grab the exact `words_to_remove` (e.g., it misses the phrase "more than 1000 employees"), those words remain in the query. The semantic model will then try to find the concept of "1000 employees" in the text, distorting the rankings.
* **Production Monitoring:** To detect these failures in production, I would monitor the difference between the Semantic score and the BM25 score for the top-ranked result. If a company has a Semantic score of 1.0 and a BM25 score of 0.0, it is a strong indicator that the engine deviated too much from the requested keywords, relying entirely on a contextual assumption (or vice versa).

---

## Critical Thinking Summary

* **Where does the system work extremely well?** It excels at "mixed" queries that combine strict constraints (location, revenue) with broad conceptual needs (e.g., "fast-growing fintech"). The `safe_filter` ensures it never fails completely.
* **How robust is the system to missing data?** By using the `candidates[column].isna()` logic, the system gives companies the "benefit of the doubt". It ensures a company isn't thrown out just because its financial data or employee count is hidden or missing. **However**, if a company is missing its actual text description or core offerings, it will sink to the bottom of the list. The system easily survives missing numbers, but it absolutely needs text to calculate the Semantic and BM25 scores.
* **What improvements would I prioritize next?** I would integrate a standard NLP library to properly remove stop words (and, the, with) from the query *after* the LLM extraction. This would make the word-count for the Dynamic Weighting logic perfectly accurate.
