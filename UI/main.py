import streamlit as st
import sys

sys.path.append("../")
from Logic import utils
import time
from enum import Enum
import random
from Logic.core.utility.snippet import Snippet
from Logic.core.link_analysis.analyzer import LinkAnalyzer
from Logic.core.indexer.index_reader import Index_reader, Indexes

snippet_obj = Snippet()


class color(Enum):
    RED = "#FF0000"
    GREEN = "#00FF00"
    BLUE = "#0000FF"
    YELLOW = "#FFFF00"
    WHITE = "#FFFFFF"
    CYAN = "#00FFFF"
    MAGENTA = "#FF00FF"


def get_top_x_movies_by_rank(x: int, results: list):
    path = "../Logic/core/indexes/"  # Link to the index folder
    document_index = Index_reader(path, Indexes.DOCUMENTS)
    corpus = []
    root_set = []
    for movie_id, movie_detail in document_index.index.items():
        movie_title = movie_detail["title"]
        stars = movie_detail["stars"]
        corpus.append({"id": movie_id, "title": movie_title, "stars": stars})

    for element in results:
        movie_id = element[0]
        movie_detail = document_index.index[movie_id]
        movie_title = movie_detail["title"]
        stars = movie_detail["stars"]
        root_set.append({"id": movie_id, "title": movie_title, "stars": stars})
    analyzer = LinkAnalyzer(root_set=root_set)
    analyzer.expand_graph(corpus=corpus)
    actors, movies = analyzer.hits(max_result=x)
    return actors, movies


def get_summary_with_snippet(movie_info, query):
    summary = movie_info["first_page_summary"]
    snippet, not_exist_words = snippet_obj.find_snippet(summary, query)
    if "***" in snippet:
        snippet = snippet.split()
        for i in range(len(snippet)):
            current_word = snippet[i]
            if current_word.startswith("***") and current_word.endswith("***"):
                current_word_without_star = current_word[3:-3]
                summary = summary.lower().replace(
                    current_word_without_star,
                    f"<b><font size='4' color={random.choice(list(color)).value}>{current_word_without_star}</font></b>",
                )
    return summary


def search_time(start, end):
    st.success("Search took: {:.2f} milliseconds".format((end - start) * 1e3))


def search_handling(
    search_button,
    search_term,
    search_max_num,
    search_weights,
    search_method,
    safe_ranking,
    unigram_smoothing,
    alpha,
    lamda,
    filter_button,
    num_filter_results,
):
    if filter_button:
        if "search_results" in st.session_state:
            top_actors, top_movies = get_top_x_movies_by_rank(
                num_filter_results, st.session_state["search_results"]
            )
            st.markdown(f"**Top {num_filter_results} Actors:**")
            actors_ = ", ".join(top_actors)
            st.markdown(
                f"<span style='color:{random.choice(list(color)).value}'>{actors_}</span>",
                unsafe_allow_html=True,
            )
            st.markdown("---")

        st.markdown(f"**Top {num_filter_results} Movies:**")
        for i in range(len(top_movies)):
            info = utils.get_movie_by_id(top_movies[i], utils.movies_dataset)
            st.markdown(f"## {i+1}. {info['title']}")
            st.write(f"### **Stars:** {', '.join(info['stars'])}")
            st.write(f'**Summary:** {info["first_page_summary"]}')

            st.markdown(
                f'<span style="color:{random.choice(list(color)).value}"><a href="{info["URL"]}">Link to movie</a></span>',
                unsafe_allow_html=True,
            )
            st.markdown("---")
        return

    if search_button:
        corrected_query = utils.correct_text(search_term, utils.all_documents)

        if corrected_query != search_term:
            st.warning(f"Your search terms were corrected to: {corrected_query}")
            search_term = corrected_query

        with st.spinner("Searching..."):
            time.sleep(0.5)  # for showing the spinner! (can be removed)
            start_time = time.time()
            result = utils.search(
                search_term,
                search_max_num,
                search_method,
                search_weights,
                unigram_smoothing=unigram_smoothing,
                alpha=alpha,
                lamda=lamda,
                safe_ranking=safe_ranking
            )
            if "search_results" in st.session_state:
                st.session_state["search_results"] = result
            end_time = time.time()
            if len(result) == 0:
                st.warning("No results found!")
                return

            search_time(start_time, end_time)

        st.markdown("<div class='grid-container'>", unsafe_allow_html=True)
        for i in range(len(result)):
            info = utils.get_movie_by_id(result[i][0], utils.movies_dataset)
            st.markdown(f"""
                            <a href="{info['URL']}" style="text-decoration: none; color: inherit;">
                            <div class='result-card'>
                                <h3>{info['title']}</h3>
                                <img src="{info['Image_URL']}" alt="Movie Image">
                                <p><b>Relevance Score:</b> {result[i][1]}</p>
                                <p><b>Summary:</b> {get_summary_with_snippet(info, search_term)}</p>
                                <p><b>Directors:</b> {', '.join(info['directors']) if info['directors'] else ''}</p>
                                <p><b>Stars:</b> {', '.join(info['stars']) if info['stars'] else ''}</p>
                                <p><b>Genres:</b> {', '.join(info['genres']) if info['genres'] else ''}</p>
                            </div>
                            </a>
                            """, unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

        st.session_state["search_results"] = result
        if "filter_state" in st.session_state:
            st.session_state["filter_state"] = (
                "search_results" in st.session_state
                and len(st.session_state["search_results"]) > 0
            )


def main():
    st.set_page_config(page_title="IMDB Movie Search Engine", layout="wide")

    st.markdown("""
        <style>
        body {
            background-color: #141414;
            color: #f5c518;
        }
        .main-header {
            text-align: center;
            margin-bottom: 1em;
        }
        .sub-header {
            font-size: 1.2em;
            color: #f5c518;
            text-align: center;
            margin-bottom: 1em;
        }
        .credits {
            color: #f5c518;
            text-align: center;
            margin-bottom: 2em;
        }
        .search-bar {
            width: 50%;
            margin: 0 auto;
        }
        .result-card {
            background-color: #555555;
            padding: 20px;
            border-radius: 10px;
            margin-bottom: 1em;
            border: 1px solid #ddd;
            box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
        }
        .result-card h3 {
            color: #333;
        }
        .result-card p, .result-card a {
            color: #333;
        }
        .result-card a {
            text-decoration: none;
            color: #f5c518;
        }
        .result-card img {
            display: block;
            margin-left: auto;
            margin-right: auto;
            margin-bottom: 1em;
            width: 100%;
            height: auto;
            max-width: 200px;
        }
        .grid-container {
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(300px, 1fr));
            gap: 1em;
            padding: 1em;
        }
        </style>
        """, unsafe_allow_html=True)

    st.markdown("""
        <div class='main-header'>
            <img src="https://upload.wikimedia.org/wikipedia/commons/6/69/IMDB_Logo_2016.svg" alt="IMDb Logo" width="200">
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<div class='sub-header'>Find the most relevant movies based on your search terms</div>",
                unsafe_allow_html=True)

    with st.container():
        search_term = st.text_input("Enter your search term here...", "", key="search")
        search_button = st.button("Search", key="search_btn")
        st.markdown("---")

    unigram_smoothing, alpha, lamda = None, None, None

    with st.expander("Advanced Search Options"):
        col1, col2, col3 = st.columns(3)

        with col1:
            weight_stars = st.slider("Weight of stars", 0.0, 1.0, 1.0, step=0.1)
            weight_genres = st.slider("Weight of genres", 0.0, 1.0, 1.0, step=0.1)
            weight_summary = st.slider("Weight of summary", 0.0, 1.0, 1.0, step=0.1)

        with col2:
            search_max_num = st.number_input("Max results", min_value=5, max_value=100, value=10, step=5)
            search_method = st.selectbox("Search method", ["ltn.lnn", "ltc.lnc", "OkapiBM25", "unigram"])
            safe_ranking = st.checkbox("Safe ranking", value=True)

        with col3:
            if search_method == "unigram":
                unigram_smoothing = st.selectbox("Smoothing method", ["naive", "bayes", "mixture"])
                if unigram_smoothing in ["bayes", "mixture"]:
                    alpha = st.slider("Alpha", 0.0, 1.0, 0.5, step=0.1)
                if unigram_smoothing == "mixture":
                    lamda = st.slider("Lambda", 0.0, 1.0, 0.5, step=0.1)

    filter_button = st.button("Filter Movies by Ranking", key="filter_btn")
    slider_ = st.selectbox("Select number of top movies to show", [1, 5, 10, 20, 50, 100])
    search_weights = [weight_stars, weight_genres, weight_summary]

    if "search_results" not in st.session_state:
        st.session_state["search_results"] = []

    search_handling(
        search_button,
        search_term,
        search_max_num,
        search_weights,
        search_method,
        safe_ranking,
        unigram_smoothing,
        alpha,
        lamda,
        filter_button,
        slider_,
    )


if __name__ == "__main__":
    main()
