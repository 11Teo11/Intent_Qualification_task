My approach evolved through several iterations to find balance between accuracy, speed, and scalability. I designed a pipeline that uses an LLM for precise upfront filtering, followed by a hybrid ranking mechanism (Semantic + Lexical) with dybamic weighting. 

The architecture includes the following components and steps:

---

**1. Data Formatting and Pre-processing**

I noticed that fields like `address`, `primary_naics`, and `secondary_naics` were stored as strings, for some companies, containing "hidden dictionaries" or lists. To parse these faster, I created a custom parsers (using `ast.literal_eval` instead of `json.loads()` since some of the strings contained single quotes `'` instead of standard `"`).
* Initial thought: To keep the code completely modular, I wanted to dynamically test every column to see if it needed to formatting.
* Issue: When testing this dynamic approach on a scaled dataset (simulating 100k elements), the processing time exploded.
* Tradeoff/Solution: I opted for a more scalable approach by hardcoding a specific list of columns that requiered formatting (`PARSE_COLS`).

---

**2. LLM Extraction and Safe Filtering**

To narrow down candidates efficiently, I used `Gemini (2.5-flash)` to extract specific criteria from the user query to act as "hard filters".
* Initial thought: Extract a wide range of criteria (industry, business model, country code, NAICS, etc.).
* Issue: This caused the hard-filtering to be too aggressive, frequently resulting in an empty list of candidates before they could even reach the next stage of the ranking system.
* Solution: I reduced the LLM's scope  extract only strict geographical data (country, region, town) and public status. (Not the best solution -> addressed later)

---

**3. Semantic Ranking and Query Cleaning**

For the ranking phase, I mapped the remaining companies using `Cosine Similarity`. The target document string was formed by concatenating `operational_name`, `description`, `core_offerings`, and `target_markets`. I encoded this and the user query using a local embedding model (`all-MiniLM-L6-v2`).
* Test Case: "Internation trade and investment company in Constanța dealing with industrial supplies" (Targeting a company named "Valbur").
* Issue: "Portul Constața" ranked 1st (52.66% matching) purely because the word "Constața" appeard in its name. Valbur ranked 2nd and there was another firm with a very close matching score, AQUA CENTER. The semantic model was heavily biased by the location name, which we had already used as a filter.
* Solution: I needed to remove the criteria already used for filtering from the query. However, the LLM translates values (e.g., extracting "ro" when the user types "Romania"). Doing a simple string replace wouldn't work. I updated the LLM prompt to explicitly return a list of the exact original phrases from the user's text so I could cleanly remove them.
* Result: After cleaning the query, "AQUA CENTER" took the first place, Valbur second, and Portul Constanta dropped to third. Valbur was close, but since the embedding model was small and local, it struggled to perfectly differentiate the complex industrial context -> so the cosine similarity wasn't enough.

---

**4. Hybrid Ranking (Adding BM25)**

To fix the context nuances, I realized I needed a way to reward exact keyword matches alongside semantic meaning.
* Solution: I introduced `BM25` (a lexical algorithm that scores exact word occurences) to complement `Cosine Similarity` (which understands context/synonyms). I normalized both scores and averaged them (50/50 split).
* Result: Valbur successfully jumped to 1st place (67.52%)

---

**5. Dynamic Weighting (The Keyword Bias)**

While the 50/50 split worked for the previous query, I tested a longer, more descriptive query.
* Test Case: "Companies that supply packaging materials for direct-to-consumer cosmetic brands".
* Issue: A company named "FLextribe" took first place with an 75.17% total score (sem: 0.50 | bm25: 1.00). It  won purely because it had 100% BM25 score (an exact keyword match for "direct-to-consumer"), even though its semantic score (50%) showed it wasn't a great contextual fit compared to companies that primarly dealt with cosmetic packaging (which had semantic scores over 65%).
* Solution: A static 50/50 split allows BM25 to easily hijack long descriptive queries if it finds a keyword match. I implemented dynamic weight based on the length of the query (excluding the words removed by the LLM):
    * <= 3 words: 30% Semantic - 70% BM25 (favors exact matches for precise, short searches).
    * <= 7 words: 50% Semantic - 50% BM25.
    *  > 7 words: 70% Semantic - 30% BM25 (favors context over keywords for long descriptions).
* Result: With dynamic weights and normalized scores, the system correctly prioritized the context. Companies that deal with cosmetic packaging in the first place dominated the top results.
  * 87.58% Shanghai Bochen Cosmetic Packaging - cncosmeticbottles.com (sem: 0.96 | bm25: 0.68)
  * 84.52% Shenzhen Itop Cosmetic Packaging - itoppacking.com (sem: 0.94 | bm25: 0.63)
  * 84.32% Flextribe - flextribe.co (sem: 0.78 | bm25: 1.00)
  * 83.79% SHEACK Packaging - sheackpkg.com (sem: 0.95 | bm25: 0.58)
The scores weren't perfect, but there was an improvement. With more research on the word limit we could get to a better result.   

---

**6. The Number Problem and Safe Fallbacks**

Even with hybrid ranking and dynamic weights, a query containing numerical constraints (e.g., "more than 1000 employees" or "revenue over $50 million") failed.
* Issue: Neither Cosine Similarity nor BM25 handles mathematical logic well.
* Initial thought: I went back to the LLM criteria extraction step and added min_employees and min_revenue to the extraction list. But now the filtering function couldn't work since the initial criteria were based on collumns names from the companies json.
* Modularity tradeoff: Since these criteria requiered mathematical operations (>=), I had to break my striclty modular filtering loop and handle them as special cases. 
* Issue: Hard-filtering on these numbers was often too harsh because many companies have missing data. If a filter was applied blindly, the candidate list would frequently become empty.
* Solution (safe_filter): I implemented a fallback mechanism. After applying each extracted criterion, if the resulting candidate list becomes empty, the system discards that specific filter and rolls back to the previous non-empty list. Furthermore, the numerical filters were adjusted to be robus to missing data by accepting missing values, ensuring companies aren't wronlgy excluded.
