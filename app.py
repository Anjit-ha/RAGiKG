import contextlib
import io
import re
from pathlib import Path
import feedparser
import streamlit as st
import streamlit.components.v1 as components
from neo4j import GraphDatabase

import free_kg_graphrag_pipeline as pipeline


st.set_page_config(
    page_title="KG + GraphRAG Pipeline",
    page_icon="🧠",
    layout="wide",
)

# -------------------------------
# 🔥 NEW: RSS FETCH FUNCTION
# -------------------------------
import feedparser
from pathlib import Path
def fetch_indian_lpg_news():
    import feedparser
    from pathlib import Path

    feeds = [
        "https://www.thehindu.com/news/feeder/default.rss",
        "https://feeds.feedburner.com/ndtvnews-top-stories",
        "https://indianexpress.com/section/india/feed/"
    ]

    # 🎯 Focused LPG + India keywords
    keywords = [
        "lpg",
        "gas cylinder",
        "cooking gas",
        "domestic gas",
        "gas subsidy",
        "ujjwala"
    ]

    save_dir = Path("Dataset_lpg/input")
    save_dir.mkdir(parents=True, exist_ok=True)

    count = 0

    for feed_url in feeds:
        feed = feedparser.parse(feed_url)

        for entry in feed.entries:
            text = (entry.title + " " + entry.summary).lower()

            # 🇮🇳 Ensure Indian context
            if "india" not in text:
                continue

            # 🔥 LPG relevance check
            if not any(k in text for k in keywords):
                continue

            content = f"""
TOPIC: LPG (India)

Title: {entry.title}
Summary: {entry.summary}
Source: {feed.feed.get("title", "Unknown")}
"""

            file_path = save_dir / f"india_lpg_news_{count}.txt"
            file_path.write_text(content.strip(), encoding="utf-8")

            count += 1

            if count >= 10:
                break

        if count >= 10:
            break

    return count


# -------------------------------
# EXISTING FUNCTIONS
# -------------------------------
@st.cache_resource
def get_driver():
    driver = GraphDatabase.driver(
        pipeline.NEO4J_URI,
        auth=(pipeline.NEO4J_USERNAME, pipeline.NEO4J_PASSWORD),
    )
    driver.verify_connectivity()
    return driver


def load_text_files(input_directory: Path) -> list[Path]:
    if not input_directory.exists():
        return []
    return sorted(
        [
            path
            for path in input_directory.iterdir()
            if path.is_file() and path.suffix.lower() == ".txt"
        ],
        key=lambda path: path.name.lower(),
    )


def capture_output(func, *args, **kwargs):
    buffer = io.StringIO()
    with contextlib.redirect_stdout(buffer):
        result = func(*args, **kwargs)
    return result, buffer.getvalue()


def cleanup_database(driver):
    with driver.session() as session:
        session.run("MATCH (n) DETACH DELETE n")


def graph_slug_for_file(file_path: Path) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", file_path.stem).strip("_")


def graph_html_path_for_file(file_path: Path) -> Path:
    graph_slug = graph_slug_for_file(file_path)
    return pipeline.BASE_DIR / "Dataset_lpg" / "graph" / f"{graph_slug}_latest.html"


def render_graph_for_file(file_path: Path) -> None:
    graph_html_path = graph_html_path_for_file(file_path)

    if graph_html_path.exists():
        components.html(
            graph_html_path.read_text(encoding="utf-8"),
            height=700,
            scrolling=True,
        )
    else:
        st.info("No graph found yet. Process file first.")


# -------------------------------
# SESSION STATE
# -------------------------------
def init_state():
    if "current_index" not in st.session_state:
        st.session_state.current_index = 0
    if "processed_files" not in st.session_state:
        st.session_state.processed_files = set()
    if "qa_history" not in st.session_state:
        st.session_state.qa_history = []
    if "show_graph" not in st.session_state:
        st.session_state.show_graph = False


init_state()

# -------------------------------
# UI
# -------------------------------
st.title("🧠 GraphRAG with LIVE News (RSS)")

with st.sidebar:
    st.header("📡 Live News Input")

    if st.button("Fetch RSS News"):
        with st.spinner("Fetching live news..."):
            count = fetch_indian_lpg_news()
            st.success(f"{count} news articles added!")

    input_dir_text = st.text_input(
        "Input folder",
        value=str(pipeline.INPUT_DIRECTORY),
    )

input_directory = Path(input_dir_text)
text_files = load_text_files(input_directory)

st.write(f"📄 Files available: {len(text_files)}")

# -------------------------------
# DB CONNECTION
# -------------------------------
driver = get_driver()

if st.button("🧹 Clear Database"):
    cleanup_database(driver)
    st.success("Database cleared!")

# -------------------------------
# PROCESS FILES
# -------------------------------
if not text_files:
    st.warning("No news files found. Fetch RSS first.")
    st.stop()

current_file = text_files[st.session_state.current_index]

st.subheader(f"📄 Current File: {current_file.name}")

with st.expander("View Content"):
    st.write(current_file.read_text())

col1, col2, col3 = st.columns(3)

# PROCESS
with col1:
    if st.button("⚙️ Process"):
        with st.spinner("Processing..."):
            pipeline.process_text_file(driver, current_file)
            st.session_state.processed_files.add(current_file)
            st.success("Processed!")

# GRAPH
with col2:
    if st.button("📊 Show Graph"):
        st.session_state.show_graph = True

# NEXT
with col3:
    if st.button("➡️ Next"):
        st.session_state.current_index += 1
        st.rerun()

# SHOW GRAPH
if st.session_state.show_graph:
    render_graph_for_file(current_file)

# -------------------------------
# Q&A
# -------------------------------
st.divider()
st.subheader("💬 Ask Questions")

question = st.text_input("Ask something")

if st.button("Ask"):
    answer, entities, _ = pipeline.answer_graphrag_query(driver, question)

    st.write("### Answer")
    st.write(answer)

    if entities:
        st.caption("Entities: " + ", ".join(entities))