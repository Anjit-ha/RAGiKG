import contextlib
import io
import re
from pathlib import Path

import streamlit as st
import streamlit.components.v1 as components
from neo4j import GraphDatabase

import free_kg_graphrag_pipeline as pipeline


st.set_page_config(
    page_title="KG + GraphRAG Pipeline",
    page_icon="",
    layout="wide",
)


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


def cleanup_database(driver) -> tuple[int, int]:
    """Delete all Neo4j nodes and relationships, matching cleanup_db.py behavior."""
    with driver.session() as session:
        session.run("MATCH (n) DETACH DELETE n")
        node_count = session.run("MATCH (n) RETURN COUNT(n) AS node_count").single()["node_count"]
        rel_count = session.run("MATCH ()-[r]-() RETURN COUNT(r) AS rel_count").single()["rel_count"]
    return node_count, rel_count


def graph_slug_for_file(file_path: Path) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", file_path.stem).strip("_") or "knowledge_graph"


def graph_image_path_for_file(file_path: Path) -> Path:
    graph_slug = graph_slug_for_file(file_path)
    return pipeline.BASE_DIR / "Dataset_lpg" / "graph" / f"{graph_slug}_latest.png"


def graph_html_path_for_file(file_path: Path) -> Path:
    graph_slug = graph_slug_for_file(file_path)
    return pipeline.BASE_DIR / "realdataset" / "graph" / f"{graph_slug}_latest.html"


def render_graph_for_file(file_path: Path) -> None:
    graph_html_path = graph_html_path_for_file(file_path)
    graph_image_path = graph_image_path_for_file(file_path)

    if graph_html_path.exists():
        components.html(
            graph_html_path.read_text(encoding="utf-8"),
            height=780,
            scrolling=True,
        )
    elif graph_image_path.exists():
        st.warning("Interactive graph not found yet. Showing static PNG instead.")
        st.image(str(graph_image_path), caption=f"Graph for {file_path.name}", use_container_width=True)
    else:
        st.info("No graph found yet. Process the file first.")


def init_state() -> None:
    defaults = {
        "current_index": 0,
        "processed_files": set(),
        "qa_history": [],
        "last_logs": "",
        "show_graph": False,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


init_state()

title_col, cleanup_col = st.columns([3, 1])
with title_col:
    st.title("GraphRAG with Incremental KG")

with st.sidebar:
    st.header("Input")
    input_dir_text = st.text_input(
        "Text folder",
        value=str(pipeline.INPUT_DIRECTORY),
    )
    input_directory = Path(input_dir_text)
    text_files = load_text_files(input_directory)

    st.metric("Files found", len(text_files))

    if st.button("Reset session", use_container_width=True):
        st.session_state.current_index = 0
        st.session_state.processed_files = set()
        st.session_state.qa_history = []
        st.session_state.last_logs = ""
        st.session_state.show_graph = False
        st.rerun()


driver = None
try:
    driver = get_driver()
except Exception as error:
    st.error(f"Neo4j connection failed: {error}")
    st.stop()

with cleanup_col:
    st.write("")
    st.write("")
    if st.button("Cleanup DB", use_container_width=True):
        with st.spinner("Cleaning Neo4j database..."):
            try:
                node_count, rel_count = cleanup_database(driver)
                st.session_state.current_index = 0
                st.session_state.processed_files = set()
                st.session_state.qa_history = []
                st.session_state.last_logs = ""
                st.session_state.show_graph = False
                st.success(f"Database cleaned. Nodes: {node_count}, relationships: {rel_count}.")
            except Exception as error:
                st.error(f"Cleanup failed: {error}")


if not text_files:
    st.warning("No .txt files found in the selected folder.")
    st.stop()

if st.session_state.current_index >= len(text_files):
    st.success("All files have been processed.")
    st.subheader("Graphs From All Processed Files")

    processed_text_files = [path for path in text_files if path in st.session_state.processed_files]
    if not processed_text_files:
        st.info("No processed graph images found in this session.")
    else:
        for row_start in range(0, len(processed_text_files), 2):
            cols = st.columns(2)
            for col, file_path in zip(cols, processed_text_files[row_start : row_start + 2]):
                graph_path = graph_image_path_for_file(file_path)
                with col:
                    st.markdown(f"**{file_path.name}**")
                    render_graph_for_file(file_path)

    if st.session_state.qa_history:
        st.divider()
        st.subheader("Q&A History")
        for item in reversed(st.session_state.qa_history):
            st.caption(f"File: {item.get('file', 'Unknown')}")
            st.markdown(f"**Question:** {item['question']}")
            st.markdown(f"**Answer:** {item['answer']}")
            if item["entities"]:
                st.caption("Retrieved entities: " + ", ".join(item["entities"]))
            st.divider()

    st.stop()

current_file = text_files[st.session_state.current_index]
is_processed = current_file in st.session_state.processed_files

progress_value = st.session_state.current_index / max(1, len(text_files))
st.progress(progress_value)

top_left, top_right = st.columns([2, 1])
with top_left:
    st.subheader(f"Current file: {current_file.name}")
with top_right:
    st.metric("Position", f"{st.session_state.current_index + 1}/{len(text_files)}")

with st.expander("File text", expanded=True):
    st.write(current_file.read_text(encoding="utf-8"))

process_col, graph_col, continue_col = st.columns([1, 1, 1])

with process_col:
    process_disabled = is_processed
    if st.button("Process this file", disabled=process_disabled, use_container_width=True):
        with st.spinner("Extracting graph, storing Neo4j data, creating embeddings..."):
            try:
                graph_updated, logs = capture_output(pipeline.process_text_file, driver, current_file)
                st.session_state.processed_files.add(current_file)
                st.session_state.last_logs = logs
                st.session_state.show_graph = False
                st.success("File processed. You can now ask GraphRAG questions.")
                st.rerun()
            except Exception as error:
                st.session_state.last_logs = ""
                st.error(f"Failed to process {current_file.name}: {error}")

with graph_col:
    if st.button("Graph", disabled=not is_processed, use_container_width=True):
        st.session_state.show_graph = not st.session_state.show_graph

with continue_col:
    continue_disabled = not is_processed
    if st.button("Continue to next file", disabled=continue_disabled, use_container_width=True):
        st.session_state.current_index += 1
        st.session_state.show_graph = False
        st.rerun()

if st.session_state.show_graph:
    graph_image_path = graph_image_path_for_file(current_file)
    graph_html_path = graph_html_path_for_file(current_file)
    if not graph_html_path.exists():
        with st.spinner("Generating graph..."):
            try:
                pipeline.export_graph_visualization(
                    driver,
                    output_dir=graph_image_path.parent,
                    graph_name=graph_slug_for_file(current_file),
                )
            except Exception as error:
                st.error(f"Graph generation failed: {error}")

    render_graph_for_file(current_file)

if st.session_state.last_logs:
    with st.expander("Processing logs", expanded=False):
        st.code(st.session_state.last_logs, language="text")

st.divider()
st.subheader("GraphRAG Q&A")

if not is_processed:
    st.info("Process the current file first, then ask questions.")
else:
    with st.form("qa_form", clear_on_submit=True):
        question = st.text_input("Ask a question about the graph")
        submitted = st.form_submit_button("Ask")

    if submitted:
        if not question.strip():
            st.warning("Enter a question first.")
        else:
            with st.spinner("Running GraphRAG..."):
                try:
                    answer, entities, context = pipeline.answer_graphrag_query(driver, question.strip())
                    st.session_state.qa_history.append(
                        {
                            "file": current_file.name,
                            "question": question.strip(),
                            "answer": answer,
                            "entities": entities,
                        }
                    )
                    st.rerun()
                except Exception as error:
                    st.error(f"GraphRAG failed: {error}")

    if st.session_state.qa_history:
        st.markdown("**Q&A History**")

    for item in reversed(st.session_state.qa_history):
        st.caption(f"File: {item.get('file', current_file.name)}")
        st.markdown(f"**Question:** {item['question']}")
        st.markdown(f"**Answer:** {item['answer']}")
        if item["entities"]:
            st.caption("Retrieved entities: " + ", ".join(item["entities"]))
        st.divider()
