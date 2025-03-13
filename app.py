import marimo

__generated_with = "0.11.17"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _():
    import marimo as mo
    import requests as rq
    from smartfunc import backend, async_backend
    from dotenv import load_dotenv
    from mosync import async_map_with_retry
    import llm
    from pydantic import BaseModel, Field, ConfigDict
    from typing import Literal


    load_dotenv(".env", override=True)
    return (
        BaseModel,
        ConfigDict,
        Field,
        Literal,
        async_backend,
        async_map_with_retry,
        backend,
        llm,
        load_dotenv,
        mo,
        rq,
    )


@app.cell(hide_code=True)
def _(rq):
    url = "https://raw.githubusercontent.com/chalda-pnuzig/emojis.json/refs/heads/master/src/list.json"

    emoji = rq.get(url).json()['emojis']
    return emoji, url


@app.cell
def _(BaseModel, async_backend, cache):
    class EmojiDescription(BaseModel):
        terms: list[str]
        description: str    
        # model_config = ConfigDict(extra='ignore')

    llmify = async_backend("claude-3.5-haiku")

    @llmify
    async def retreive(e) -> EmojiDescription:
        """What can you tell me about this emoji {{e}}? Make sure that the description is about 2-3 sentences and that your return at least 4 terms or phrases that help describe how the emoji might be used."""

    async def get_info(e):
        if not e["emoji"] in cache:
            resp = await retreive(e['emoji'])
            cache[e['emoji']] = {**e, 'response': resp}
        return cache[e['emoji']]
    return EmojiDescription, get_info, llmify, retreive


@app.cell
async def _(get_info):
    await get_info({"emoji": '😶'})
    return


@app.cell
def _():
    from diskcache import Cache

    cache = Cache("emojidb")
    return Cache, cache


@app.cell
def _(cache, emoji):
    todo = [e for e in emoji if e['emoji'] not in cache][:10]
    len(todo), len(cache), len(emoji)
    return (todo,)


@app.cell
async def _(async_map_with_retry, get_info, todo):
    results = await async_map_with_retry(
        items=todo,
        func=get_info,
        max_concurrency=8,
        max_retries=3,
        show_progress=True
    )
    return (results,)


@app.cell
def _(cache):
    import polars as pl 
    from lazylines import LazyLines

    pl.DataFrame(
        LazyLines([cache[k] for k in cache.iterkeys()])
          .mutate(
              desc=lambda d: d['response']['description'],
              terms=lambda d: d['response']['terms']
          )
          .drop("response")
          .show()
    )
    return LazyLines, pl


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
