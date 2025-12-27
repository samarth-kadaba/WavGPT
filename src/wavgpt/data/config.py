DATASET_CONFIGS = {
    # === RECOMMENDED: C4 - Large diverse web corpus ===
    "c4": {
        "path": "allenai/c4",
        "name": "en",
        "split": "train",
        "text_field": "text",
        "description": "C4 (Colossal Clean Crawled Corpus) - 750GB diverse web text",
    },
    # === SMALL & FAST: WikiText ===
    "wikitext": {
        "path": "wikitext",
        "name": "wikitext-103-raw-v1",
        "split": "train",
        "text_field": "text",
        "description": "WikiText-103 - Wikipedia articles (good for testing)",
    },
    # === LONG CONTEXT: Wikipedia full articles ===
    "wikipedia": {
        "path": "wikipedia",
        "name": "20220301.en",
        "split": "train",
        "text_field": "text",
        "description": "Wikipedia - Full English Wikipedia articles",
    },
    # === BOOKS: Gutenberg (try parquet version) ===
    "gutenberg": {
        "path": "sedthh/gutenberg_english",
        "split": "train",
        "text_field": "TEXT",
        "description": "Gutenberg books - Long-form literature",
    },
    # === CODE: The Stack ===
    "code": {
        "path": "bigcode/starcoderdata",
        "split": "train",
        "text_field": "content",
        "description": "StarCoder data - Diverse code corpus",
    },
    # === ACADEMIC: ArXiv abstracts ===
    "arxiv": {
        "path": "ccdv/arxiv-summarization",
        "split": "train",
        "text_field": "article",
        "description": "ArXiv papers - Scientific articles",
    },
}
