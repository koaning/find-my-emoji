import marimo

__generated_with = "0.11.0"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import requests as rq
    return mo, rq


@app.cell
def _(rq):
    url = "https://raw.githubusercontent.com/chalda-pnuzig/emojis.json/refs/heads/master/src/list.json"

    emoji = rq.get(url).json()['emojis']
    return emoji, url


@app.cell
def _(emoji):
    emoji[0]
    return


@app.cell
def _():
    from instructor.batch import BatchJob
    from pydantic import BaseModel, Field
    from typing import Literal

    class EmojiDescription(BaseModel):
        terms: list[str] = Field(..., description="List of words/phrases that could fit the emoji. List can be long, around 10 examples.")
        description: str = Field(..., description="Describes the emoji at length. It does not describe what it looks like, but rather what the symbol could mean and what it is typically used for. Two, max three, sentences.")
    return BaseModel, BatchJob, EmojiDescription, Field, Literal


@app.cell
def _():
    return


@app.cell
def _(EmojiDescription, cache):
    import llm

    model = llm.get_async_model("claude-3.5-haiku")

    async def get_info(e):
        resp = await model.prompt(f"What can you tell me about this emoji: {e['emoji']}", schema=EmojiDescription)
        cache[e['emoji']] = {**e, 'response': resp.response_json}
        return cache[e['emoji']]
    return get_info, llm, model


@app.cell
def _():
    from diskcache import Cache

    cache = Cache("emojidb")
    return Cache, cache


@app.cell
def _(cache):
    len(cache)
    return


@app.cell
def _(cache, emoji):
    todo = [e for e in emoji if e['emoji'] not in cache][:25]
    len(todo)
    return (todo,)


@app.cell
def _(results):
    results
    return


@app.cell
async def _(async_map_with_retry, get_info, todo):
    results = await async_map_with_retry(
        items=todo,
        func=get_info,
        max_concurrency=5,
        max_retries=3,
        show_progress=True
    )
    return (results,)


@app.cell
def _(mo):
    import asyncio
    import random
    import logging
    from functools import wraps
    import tqdm
    from typing import List, Dict, Any, Callable, Optional

    async def process_with_retry(
        func, 
        item, 
        max_retries=3, 
        initial_backoff=1.0, 
        backoff_factor=2.0, 
        jitter=0.1,
        timeout=None,
        on_success=None,
        on_failure=None,
        logger=None
    ):
        """Process a single item with retry logic and backoff."""
        logger = logger or logging.getLogger(__name__)
        attempts = 0
        last_exception = None

        while attempts <= max_retries:
            try:
                # Add timeout if specified
                if timeout is not None:
                    result = await asyncio.wait_for(func(item), timeout=timeout)
                else:
                    result = await func(item)

                # Call success callback if provided
                if on_success:
                    on_success(item, result)

                return item, result, None

            except Exception as e:
                attempts += 1
                last_exception = e

                if attempts <= max_retries:
                    # Calculate backoff time with jitter
                    backoff_time = initial_backoff * (backoff_factor ** (attempts - 1))
                    jitter_amount = backoff_time * jitter
                    actual_backoff = backoff_time + random.uniform(-jitter_amount, jitter_amount)
                    actual_backoff = max(0.1, actual_backoff)  # Ensure minimum backoff

                    logger.warning(
                        f"Attempt {attempts}/{max_retries} failed for item {item}. "
                        f"Retrying in {actual_backoff:.2f}s. Error: {str(e)}"
                    )

                    await asyncio.sleep(actual_backoff)
                else:
                    if on_failure:
                        on_failure(item, last_exception)

                    logger.error(
                        f"All {max_retries} retry attempts failed for item {item}. "
                        f"Final error: {str(last_exception)}"
                    )

        return item, None, last_exception

    async def async_map_worker(
        items, 
        func, 
        semaphore,
        max_retries=3, 
        initial_backoff=1.0, 
        backoff_factor=2.0, 
        jitter=0.1,
        timeout=None,
        on_success=None,
        on_failure=None,
        logger=None
    ):
        """Map an async function over items with concurrency control."""
        async def bounded_process(item):
            async with semaphore:
                return await process_with_retry(
                    func, 
                    item, 
                    max_retries, 
                    initial_backoff, 
                    backoff_factor, 
                    jitter,
                    timeout,
                    on_success,
                    on_failure,
                    logger
                )

        # Create tasks
        tasks = [bounded_process(item) for item in items]
        return tasks

    def async_map_with_retry(
        items: List[Dict[Any, Any]],
        func: Callable,
        max_concurrency: int = 10,
        max_retries: int = 3,
        initial_backoff: float = 1.0,
        backoff_factor: float = 2.0,
        jitter: float = 0.1,
        timeout: Optional[float] = None,
        on_success: Optional[Callable] = None,
        on_failure: Optional[Callable] = None,
        show_progress: bool = True,
        description: str = "Processing items",
        logger: Optional[logging.Logger] = None
    ):
        """
        Map an async function over a list of dictionaries with progress tracking and retry.

        Args:
            items: List of dictionaries to process
            func: Async function that takes a dictionary and returns a result
            max_concurrency: Maximum number of concurrent tasks
            max_retries: Maximum number of retry attempts
            initial_backoff: Initial backoff time in seconds
            backoff_factor: Multiplier for backoff on successive retries
            jitter: Random jitter factor to avoid thundering herd
            timeout: Maximum time to wait for a task to complete (None = wait forever)
            on_success: Callback function to run on successful processing
            on_failure: Callback function to run when an item fails after all retries
            show_progress: Whether to show progress bar
            description: Description for progress bar
            logger: Optional logger for detailed logging

        Returns:
            List of tuples (original_dict, result_or_None, exception_or_None)
        """
        logger = logger or logging.getLogger(__name__)

        async def main():
            # Create semaphore for concurrency control
            semaphore = asyncio.Semaphore(max_concurrency)

            # Get tasks
            tasks = await async_map_worker(
                items, 
                func, 
                semaphore,
                max_retries, 
                initial_backoff, 
                backoff_factor, 
                jitter,
                timeout,
                on_success,
                on_failure,
                logger
            )

            # Set up progress bar if requested
            if show_progress:
                results = []
                with mo.status.progress_bar(total=len(tasks), title=description) as progress_bar:
                    for task in asyncio.as_completed(tasks):
                        result = await task
                        results.append(result)
                        progress_bar.update()
                return results
            else:
                # Without progress bar, just gather all results
                return await asyncio.gather(*tasks)

        return main()
    return (
        Any,
        Callable,
        Dict,
        List,
        Optional,
        async_map_with_retry,
        async_map_worker,
        asyncio,
        logging,
        process_with_retry,
        random,
        tqdm,
        wraps,
    )


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
