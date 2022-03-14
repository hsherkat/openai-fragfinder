from flask import Flask, redirect, render_template, request, url_for

import config
from recommendations import (
    df_reviews_and_embeddings,
    fragrance_embeddings,
    fragrance_recommender,
    fragrance_recommender_alt,
)

app = Flask(__name__)


@app.route("/", methods=("GET", "POST"))
def index():
    """Shut up pylint.
    """
    if request.method == "POST":
        frag_query = request.form["frag"]
        recs = fragrance_recommender_alt(frag_query, fragrance_embeddings)
        print(recs)
        response = recs.to_html(index=False)
        return redirect(url_for("index", result=response, query=frag_query))

    result = request.args.get("result")
    query = request.args.get("query")
    return render_template("index.html", result=result, query=query)


if __name__ == "__main__":
    app.run()

# use the code below to use original fragrance recommender

# @app.route("/", methods=("GET", "POST"))
# def index():
#     """Shut up pylint.
#     """
#     if request.method == "POST":
#         frag_query = request.form["frag"]
#         recs = fragrance_recommender(frag_query, df_reviews_and_embeddings)
#         formatted_recs = recs.copy()
#         print(recs)
#         formatted_recs["Fragrance"] = recs.apply(
#             lambda row: f"{row['brand']}: {row['name']}", axis=1
#         )
#         formatted_recs.rename(
#             columns={"cleaned_review": "Relevant review"}, inplace=True
#         )
#         response = formatted_recs[["Fragrance", "Relevant review"]].to_html(index=False)
#         return redirect(url_for("index", result=response,query=frag_query))

#     result = request.args.get("result")
#     query = request.args.get("query")
#     return render_template("index.html", result=result,query=query)


# use the code below to use alternate fragrance recommender

# @app.route("/", methods=("GET", "POST"))
# def index():
#     """Shut up pylint.
#     """
#     if request.method == "POST":
#         frag_query = request.form["frag"]
#         recs = fragrance_recommender_alt(frag_query, fragrance_embeddings)
#         print(recs)
#         response = recs.to_html(index=False)
#         return redirect(url_for("index", result=response, query=frag_query))

#     result = request.args.get("result")
#     query = request.args.get("query")
#     return render_template("index.html", result=result, query=query)
