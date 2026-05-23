"""
BLEU Score Calculator for GraphRAG Outputs — Real Dataset
==========================================================
Compares answers produced across multiple files for the same query.
Each consecutive pair is scored: file N answer vs file N+1 answer.

Install dependency:
pip install nltk

Usage:
python bleu_score_calculator_real.py
"""

from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction


def compute_bleu(reference: str, hypothesis: str) -> float:
# Simple whitespace tokenisation — no punkt download needed
ref_tokens = reference.lower().split()
hyp_tokens = hypothesis.lower().split()
smoother = SmoothingFunction().method1
score = sentence_bleu(
references=[ref_tokens],
hypothesis=hyp_tokens,
weights=(0.25, 0.25, 0.25, 0.25),
smoothing_function=smoother,
)
return round(score, 4)


def evaluate_query(query_name: str, answers: dict) -> list:
print(f"\n{'='*65}")
print(f"Query: {query_name}")
print(f"{'='*65}")

labels = list(answers.keys())
texts = list(answers.values())

print(f"{'File Pair':<18} {'BLEU Score':>12} Interpretation")
    print("-" * 65)

    scores = []
    for i in range(len(texts) - 1):
    ref = texts[i]
    hyp = texts[i + 1]
    score = compute_bleu(ref, hyp)
    scores.append(score)

    if score == 0.0:
    interpretation = "Total change — no shared phrases"
    elif score < 0.30: interpretation="Major change — significant new info added" elif score < 0.60:
        interpretation="Moderate change — partial overlap" elif score < 0.90:
        interpretation="Minor change — mostly same, small addition" else: interpretation="Identical or near-identical"
        print(f"{labels[i]+' -> '+labels[i+1]:<18} {score:>12.4f} {interpretation}")

            avg = sum(scores) / len(scores) if scores else 0
            print("-" * 65)
            print(f"{'Average BLEU':<18} {avg:>12.4f}")
                return scores


                # ─────────────────────────────────────────────────────────────────
                # YOUR DATA — Real Dataset answers
                # ─────────────────────────────────────────────────────────────────

                # Query 1: "Which companies committed to carbon neutrality and by when?"
                query1_answers = {
                "file1": (
                "Not found in the graph."
                ),
                "file2": (
                "Google achieved Carbon Neutrality, no specific date mentioned. "
                "Microsoft committed to Carbon Negative Pledge, no specific date mentioned."
                ),
                "file3": (
                "Google achieved Carbon Neutrality, no specific date mentioned. "
                "Microsoft committed to Carbon Negative Pledge, no specific date mentioned."
                ),
                "file4": (
                "Google achieved Carbon Neutrality, Microsoft committed to Carbon Negative Pledge."
                ),
                "file5": (
                "Google achieved Carbon Neutrality, no specific date mentioned. "
                "Microsoft committed to Carbon Negative Pledge, no specific date mentioned. "
                "Blackrock, Vanguard, and Fidelity joined Net Zero Asset Managers, "
                "no specific date mentioned."
                ),
                }

                # Query 2: "What is the IPCC and who does it work with?"
                query2_answers = {
                "file1": (
                "Ipcc is an entity that studied Climate Change, established by United Nations "
                "Environment Programme and World Meteorological Organization. It released the "
                "Sixth Assessment Report and works with Greenpeace, which lobbied Ipcc."
                ),
                "file2": (
                "Ipcc is an entity that studied Climate Change, established by United Nations "
                "Environment Programme and World Meteorological Organization. "
                "It works with International Energy Agency."
                ),
                "file3": (
                "Ipcc is an entity that studied Climate Change, established by United Nations "
                "Environment Programme and World Meteorological Organization. It works with "
                "International Energy Agency and has partnerships with World Bank."
                ),
                "file4": (
                "Ipcc is an entity that studied Climate Change, established by United Nations "
                "Environment Programme and World Meteorological Organization. "
                "It works with International Energy Agency."
                ),
                "file5": (
                "Ipcc is an entity that studied Climate Change, established by United Nations "
                "Environment Programme and World Meteorological Organization. It works with "
                "International Energy Agency and has a partnership with World Bank."
                ),
                }

                # ─────────────────────────────────────────────────────────────────
                # Run evaluation
                # ─────────────────────────────────────────────────────────────────
                if __name__ == "__main__":
                print("\nGraphRAG BLEU Score Evaluation — Real Dataset (Climate Change)")
                print("=" * 65)
                print("Method : sentence_bleu (4-gram) with smoothing (method1)")
                print("Pairs : consecutive file answers for each query")

                scores_q1 = evaluate_query(
                "Which companies committed to carbon neutrality and by when?",
                query1_answers
                )
                scores_q2 = evaluate_query(
                "What is the IPCC and who does it work with?",
                query2_answers
                )

                all_scores = scores_q1 + scores_q2
                overall_avg = sum(all_scores) / len(all_scores)

                print(f"\n{'='*65}")
                print(f"OVERALL AVERAGE BLEU (both queries, all pairs): {overall_avg:.4f}")
                print(f"{'='*65}")

                print("""
                HOW TO ADD MORE QUERIES
                -----------------------
                1. Create a new dict with file1..file5 keys and your answers as values.
                2. Call evaluate_query('your question', your_dict) below the existing calls.
                3. Add the returned scores list to all_scores for the overall average.

                Example:
                query3_answers = {
                "file1": "answer from file 1...",
                "file2": "answer from file 2...",
                ...
                }
                scores_q3 = evaluate_query("Your question here?", query3_answers)
                all_scores += scores_q3
                """)