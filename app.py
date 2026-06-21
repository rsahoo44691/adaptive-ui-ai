"""Adaptive Content Dashboard -- an AI-based adaptive user interface.

The interface starts in a neutral, editorial state. As the user likes, skips,
or reads articles, an online machine-learning model (see adaptive_engine.py)
updates in real time and the UI adapts at runtime:

  * the "For You" feed re-ranks itself toward the user's learned interests;
  * a live preference panel shows which sections the UI is surfacing;
  * a "Why am I seeing this?" panel keeps the adaptation transparent;
  * an accessibility focus reading mode turns on automatically when the user
    shows a preference for long-form reading;
  * privacy controls let the user pause learning or erase what was learned.

Run with:  python app.py
Then open the local URL printed in the terminal.
"""

import gradio as gr

from adaptive_engine import AdaptiveEngine
from content import get_articles

# --------------------------------------------------------------------------- #
# Rendering helpers
# --------------------------------------------------------------------------- #

def _feed_html(engine):
    """Render the ranked feed as HTML, scaling type for reading mode."""
    font = "1.18rem" if engine.reading_mode else "1.0rem"
    line = "1.9" if engine.reading_mode else "1.55"
    pad = "20px" if engine.reading_mode else "14px"

    rows = []
    for rank, item in enumerate(engine.ranked(), start=1):
        pct = int(item["score"] * 100)
        seen_tag = (" &nbsp;<span style='color:#2563eb;'>&#10003; seen</span>"
                    if item["id"] in engine.seen else "")
        long_tag = (" &nbsp;<span style='background:#fef3c7;color:#92400e;"
                    "padding:1px 6px;border-radius:6px;font-size:.8em;'>"
                    "long read</span>" if item["reading_time"] >= 6 else "")
        rows.append(f"""
        <div style="border:1px solid #e5e7eb;border-radius:12px;padding:{pad};
                    margin-bottom:12px;background:#ffffff;line-height:{line};">
          <div style="display:flex;justify-content:space-between;
                      align-items:baseline;">
            <span style="font-weight:600;font-size:{font};">#{rank}
              &nbsp;{item['title']}</span>
            <span style="color:#6b7280;font-size:.85em;">match {pct}%</span>
          </div>
          <div style="color:#6b7280;font-size:.85em;margin:4px 0;">
            {item['category']} &middot; {item['reading_time']} min{long_tag}{seen_tag}
          </div>
          <div style="height:6px;background:#f3f4f6;border-radius:6px;
                      overflow:hidden;margin:6px 0;">
            <div style="height:100%;width:{pct}%;background:#3b82f6;"></div>
          </div>
          <div style="font-size:{font};margin:6px 0;">{item['body']}</div>
          <div style="color:#2563eb;font-size:.85em;font-style:italic;">
            &#128161; {item['reason']}
          </div>
        </div>""")
    banner = ""
    if engine.reading_mode:
        banner = ("<div style='background:#ecfdf5;border:1px solid #a7f3d0;"
                  "color:#065f46;padding:8px 12px;border-radius:8px;"
                  "margin-bottom:12px;'>&#9855; Focus reading mode is on -- "
                  "text is enlarged for comfortable long-form reading.</div>")
    return f"<div>{banner}{''.join(rows)}</div>"


def _preferences_html(engine):
    """Live panel showing the learned preference model."""
    aff = engine.affinity_summary()
    if not aff:
        body = ("<p style='color:#6b7280;'>No preferences learned yet. "
                "Like or read a few articles and watch this panel adapt.</p>")
    else:
        bars = []
        peak = max(abs(s) for _, s in aff) or 1
        for cat, score in aff:
            width = int(abs(score) / peak * 100)
            color = "#10b981" if score >= 0 else "#ef4444"
            label = "+" if score >= 0 else "-"
            bars.append(f"""
            <div style="margin-bottom:8px;">
              <div style="font-size:.9em;">{cat}
                <span style="color:#6b7280;">({label}{abs(round(score,1))})</span>
              </div>
              <div style="height:8px;background:#f3f4f6;border-radius:6px;">
                <div style="height:100%;width:{width}%;background:{color};
                            border-radius:6px;"></div>
              </div>
            </div>""")
        body = "".join(bars)
    return (f"<div><h4 style='margin:0 0 8px;'>What the UI has learned</h4>"
            f"<p style='color:#6b7280;font-size:.85em;margin:0 0 10px;'>"
            f"{engine.interactions} interaction(s) observed</p>{body}</div>")


def _log_html(engine):
    if not engine.log:
        return ("<p style='color:#6b7280;'>Adaptation history will appear "
                "here as you interact.</p>")
    items = "".join(
        f"<li style='margin-bottom:4px;font-size:.88em;'>{m}</li>"
        for m in reversed(engine.log[-12:])
    )
    return f"<ul style='padding-left:18px;margin:0;'>{items}</ul>"


def _choices(engine):
    return [(f"{a['title']}  ({a['category']})", a["id"])
            for a in engine.articles]


def _render(engine):
    """Bundle every dynamic output after a state change."""
    return (_feed_html(engine), _preferences_html(engine), _log_html(engine))


# --------------------------------------------------------------------------- #
# Event handlers (each takes the per-session engine from gr.State)
# --------------------------------------------------------------------------- #

def on_action(engine, article_id, action):
    if not article_id:
        status = "Select an article first."
    else:
        status = engine.observe(article_id, action)
    feed, prefs, log = _render(engine)
    return engine, feed, prefs, log, status


def on_toggle_learning(engine, enabled):
    engine.set_learning(enabled)
    feed, prefs, log = _render(engine)
    state = "enabled" if enabled else "paused"
    return engine, feed, prefs, log, f"Adaptive learning {state}."


def on_toggle_reading(engine, enabled):
    engine.set_reading_mode(enabled)
    feed, prefs, log = _render(engine)
    return engine, feed, prefs, log, ""


def on_reset(engine):
    engine.reset()
    feed, prefs, log = _render(engine)
    return engine, feed, prefs, log, "Profile reset. The UI is neutral again."


def on_export(engine):
    return engine.export_profile()


# --------------------------------------------------------------------------- #
# UI definition
# --------------------------------------------------------------------------- #

def build_app():
    seed = AdaptiveEngine(get_articles())

    with gr.Blocks(title="Adaptive Content Dashboard",
                   theme=gr.themes.Soft()) as demo:
        engine_state = gr.State(seed)

        gr.Markdown(
            "# &#129504; Adaptive Content Dashboard\n"
            "An **AI-based adaptive user interface**. The feed below learns "
            "from what you like, skip, and read, then re-ranks itself and "
            "adapts its presentation in real time."
        )

        with gr.Row():
            # ---------------- main feed column ----------------
            with gr.Column(scale=3):
                gr.Markdown("### &#128241; Your Feed")
                feed_out = gr.HTML(_feed_html(seed))

            # ---------------- control / insight column ----------------
            with gr.Column(scale=2):
                gr.Markdown("### &#9881;&#65039; Interact")
                article_dd = gr.Dropdown(
                    choices=_choices(seed),
                    label="Pick an article to act on",
                    value=None,
                )
                with gr.Row():
                    like_btn = gr.Button("&#128077; Like", variant="primary")
                    read_btn = gr.Button("&#128214; Read")
                    skip_btn = gr.Button("&#128078; Skip")
                status_out = gr.Markdown("")

                gr.Markdown("### &#128202; Adaptive Insights")
                prefs_out = gr.HTML(_preferences_html(seed))

                with gr.Accordion("Why am I seeing this? (transparency)",
                                  open=False):
                    log_out = gr.HTML(_log_html(seed))

                with gr.Accordion("Accessibility & privacy controls",
                                  open=False):
                    reading_chk = gr.Checkbox(
                        label="Focus reading mode (larger text)",
                        value=False,
                    )
                    learning_chk = gr.Checkbox(
                        label="Adaptive learning on", value=True,
                    )
                    reset_btn = gr.Button("Reset my data")
                    export_btn = gr.Button("Show data stored about me")
                    export_out = gr.Textbox(
                        label="Your stored profile", lines=8,
                        interactive=False, visible=True,
                    )

        outputs = [engine_state, feed_out, prefs_out, log_out, status_out]

        like_btn.click(lambda e, a: on_action(e, a, "like"),
                       [engine_state, article_dd], outputs)
        read_btn.click(lambda e, a: on_action(e, a, "read"),
                       [engine_state, article_dd], outputs)
        skip_btn.click(lambda e, a: on_action(e, a, "skip"),
                       [engine_state, article_dd], outputs)

        reading_chk.change(on_toggle_reading,
                           [engine_state, reading_chk], outputs)
        learning_chk.change(on_toggle_learning,
                            [engine_state, learning_chk], outputs)
        reset_btn.click(on_reset, [engine_state], outputs)
        export_btn.click(on_export, [engine_state], [export_out])

    return demo


if __name__ == "__main__":
    build_app().launch()
