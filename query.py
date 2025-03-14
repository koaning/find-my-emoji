import marimo

__generated_with = "0.11.17"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import srsly
    import polars as pl
    from lazylines import LazyLines
    return LazyLines, mo, pl, srsly


@app.cell
def _(pl, srsly):
    df = (
        pl.DataFrame(srsly.read_json("data/download.json"))
          .select("emoji", "terms")
          .explode("terms")
          .with_row_index()
    )
    return (df,)


@app.cell
def _():
    from model2vec import StaticModel

    # Load a model from the HuggingFace hub (in this case the potion-base-8M model)
    model = StaticModel.from_pretrained("minishlab/potion-base-32M")
    return StaticModel, model


@app.cell
def _(df, model, text_ui):
    from simsity import create_index, load_index

    # Populate the ANN vector index and use it. 
    index = create_index(df.to_dicts(), lambda d: model.encode([_["terms"] for _ in d]))
    items, dists = index.query({"terms": text_ui.value}, k=20)
    return create_index, dists, index, items, load_index


@app.cell
def _(mo):
    text_ui = mo.ui.text(label="Let's search for emoji!")
    text_ui
    return (text_ui,)


@app.cell
def _(items, mo, p, text_ui):
    mo.stop(text_ui.value == "", "Give a search query")

    p("".join(set([_["emoji"] for _ in items])), style="font-size: 65px;")
    return


@app.cell
def _():
    from mohtml import p, div, tailwind_css, br

    tailwind_css()
    return br, div, p, tailwind_css


@app.cell
def _(dists, items, pl):
    pl.DataFrame(items).with_columns(dist=dists)
    return


@app.cell
def _(br, buttonstack, div, get_example, p, pl, srsly):
    emoji_dict = pl.DataFrame(srsly.read_json("data/download.json")).select("emoji", "desc").to_dicts()
    mapping = { _["emoji"]: _["desc"] for _ in emoji_dict }
    ex = get_example()
    div(
        p(
            f"{ex['emoji']} = {ex['terms']}", 
            klass="text-7xl font-bold p-2"
         ),
        p(mapping[ex['emoji']]),
        br(),
        buttonstack,
        klass="p-8 bg-gray-800 rounded-lg",
    )
    return emoji_dict, ex, mapping


@app.cell
def _(gen, get_example, get_labels, set_example, set_labels):
    def add_label(lab):
        new_state = get_labels() + [{"example": get_example(), "annotation": lab}]
        set_labels(new_state)
        set_example(next(gen))

    def undo():
        set_labels(get_labels()[:-2])
    return add_label, undo


@app.cell
def _(gen, mo):
    get_labels, set_labels = mo.state([])
    get_example, set_example = mo.state(next(gen))
    return get_example, get_labels, set_example, set_labels


@app.cell
def _(add_label, mo, undo):
    btn_yes  = mo.ui.button(value=0, label=f"yes - j", keyboard_shortcut=f"Ctrl-j", on_click=lambda d: d + 1, on_change=lambda d: add_label("yes")) 
    btn_no   = mo.ui.button(value=0, label=f"no - k", keyboard_shortcut=f"Ctrl-k", on_click=lambda d: d + 1, on_change=lambda d: add_label("no")) 
    btn_skip = mo.ui.button(value=0, label=f"skip - l", keyboard_shortcut=f"Ctrl-l", on_click=lambda d: d + 1, on_change=lambda d: add_label("skip")) 
    btn_undo = mo.ui.button(value=0, label=f"undo - ;", keyboard_shortcut=f"Ctrl-;", on_click=lambda d: d + 1, on_change=lambda d: undo()) 

    buttonstack = mo.hstack([btn_yes, btn_no, btn_skip, btn_undo])
    return btn_no, btn_skip, btn_undo, btn_yes, buttonstack


@app.cell
def _(get_labels, pl):
    pl.DataFrame(get_labels()).reverse()
    return


@app.cell
def _(df):
    gen = (ex for ex in df.sample(1000).to_dicts())
    return (gen,)


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
