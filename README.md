This is a fragrance recommendation app powered by OpenAI's API.

Its data consists of reviews of fragrances from the 24 most popular brands, with 40 fragrances from each brand.
Reviews were scraped from Basenotes, a site dedicated to fragrances, using BeautifulSoup. Fragrance note data was also scraped, but currently not used for making recommendations. A more sophisticated recommender could use both reviews and note data.

There are recommender methods that can be used. Both work by using OpenAI's API to calculate embeddings of the reviews, as well as an embedding of a fragrance description (the query). Then the query embedding can be compared to individual review embeddings (using cosine similarity), or it can be compared to the mean of the review embeddings for a fragrance (the fragrance embedding), returning the five most similar items. The former will look more impressive, since the reviews will match the text of the query better, though the latter may possibly give more accurate recommendations. 