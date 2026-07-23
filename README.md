# Wikipedia Speed Run Bot

A bot that tries to navigate from one Wikipedia article to another by
following links, using sentence embeddings to greedily pick the link
whose text is most semantically similar to the target page at each step.

## How it works

1. Scrape all article links (text + URL) from the current page.
2. Encode the target page name and every candidate link text with
   [`all-MiniLM-L6-v2`](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2).
3. Follow the link with the highest cosine similarity to the target.
4. Repeat until the target page is reached, a page has no unvisited
   links (dead end), or 50 hops are exceeded (give up).

## Setup

```bash
pip install requests beautifulsoup4 sentence-transformers
```

## Usage

```bash
python bot.py "Start Page" "Target Page"
```

Example:

```bash
python bot.py "Erling Haaland" "Pikachu"
```

The bot prints each page it visits, then reports success/failure and
the number of hops taken.
