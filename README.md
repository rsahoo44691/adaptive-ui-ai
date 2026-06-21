# Adaptive Content Dashboard

An **AI-based adaptive user interface** built with Python and
[Gradio](https://www.gradio.app/). The dashboard presents a feed of articles
and adapts to the user **at runtime**: it learns preferences from likes,
reads, and skips, re-ranks the feed with a content-based recommender, surfaces
the sections the user cares about, and switches on an accessibility-friendly
focus reading mode when it detects a preference for long-form content.

Built for the MSAI-631 *Artificial Intelligence for Human-Computer
Interaction* Week 7 "Adaptive UI using AI" project.

## What makes it "adaptive" and "AI-based"

| Layer | Technique |
|-------|-----------|
| Content representation | TF-IDF vectors over the article corpus |
| User model | Online preference vector updated incrementally after every interaction |
| Recommendation | Cosine similarity (relevance) + exploration term to avoid a filter bubble |
| UI adaptation | Live feed re-ranking, preference panel, auto focus-reading mode |
| Transparency | Per-item "why am I seeing this?" explanations + adaptation log |
| Privacy | Pause-learning toggle, reset/erase data, view stored profile |

The "intelligent" decisions (what to show and how to present it) are driven by
the model in `adaptive_engine.py`, making this an **Intelligent User
Interface** in service of an adaptive one.

## Project layout

```
adaptive-ui-ai/
├── app.py               # Gradio UI + event wiring
├── adaptive_engine.py   # TF-IDF model + online user-preference learning
├── content.py           # Synthetic article corpus
├── requirements.txt
└── README.md
```

## Run it

```bash
python3 -m venv .venv && source .venv/bin/activate   # optional
pip install -r requirements.txt
python app.py
```

Open the local URL printed in the terminal (default
`http://127.0.0.1:7860`).

## How to demo the adaptation

1. Like or **Read** two AI & Machine Learning articles. The feed re-orders to
   put related AI items on top, and the preference panel shows a green bar for
   that section.
2. Open a couple of *long read* articles — the UI automatically enables
   **focus reading mode** and enlarges the text (accessibility adaptation).
3. Open **"Why am I seeing this?"** to read the per-item explanations and the
   adaptation history.
4. Use the **privacy controls** to pause learning or reset your profile, then
   watch the feed return to its neutral editorial order.

## AI tool disclosure

An AI coding assistant was used to help scaffold and document this project.
All generated code was reviewed and tested by the author.
