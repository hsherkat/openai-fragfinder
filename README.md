This is a fragrance recommendation app powered by OpenAI's API.

The data it relies on consists of reviews of fragrances from the 24 most popular brands, with 40 fragrances from each brand.
Reviews were scraped from Basenotes, a site dedicated to fragrances, using aiohttp (to asynchronously scrape) and BeautifulSoup. Fragrance note data was also scraped, but currently is not used for making recommendations. A more sophisticated recommender could use both reviews and note data.

There are two recommender functions that can be used. Both work by using OpenAI's API to calculate embeddings of the reviews, as well as an embedding of a fragrance description (the query). Then the query embedding can be compared (using cosine similarity) to individual review embeddings, or it can be compared to each fragrance's mean review embedding, returning the five most similar items. The former will look more impressive, since the reviews will match the text of the query better, though the latter may possibly give more accurate recommendations.

The website is hacked together with help from OpenAI's animal superhero naming app example in the documentation. (The dog icon is too cute; I had to include it.)

Note:
The data files are not included, as they are too large. If cloned, this can not be run without scraping the fragrance data and generating the review emeddings.

Some screenshots:

![home page](https://github.com/hsherkat/openai-fragfinder/blob/master/screenshots/screenshot0.png)

![sample](https://github.com/hsherkat/openai-fragfinder/blob/master/screenshots/screenshot1.png)

![sample](https://github.com/hsherkat/openai-fragfinder/blob/master/screenshots/screenshot2.png)

![sample](https://github.com/hsherkat/openai-fragfinder/blob/master/screenshots/screenshot%20dustin.png)

![sample](https://github.com/hsherkat/openai-fragfinder/blob/master/screenshots/screenshot%20eric.png)
