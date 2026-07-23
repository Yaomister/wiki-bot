import time
import requests
import argparse
from bs4 import BeautifulSoup
from sentence_transformers import SentenceTransformer, util



headers = {
    "User-Agent": "wiki-bot/1.0"
}

model = SentenceTransformer("all-MiniLM-L6-v2")

def _scrape(current_link):
    """Fetch a Wikipedia page and get the link for every article referenced."""

    html = requests.get(current_link, headers=headers).text
    soup = BeautifulSoup(html, "html.parser")

    body = soup.find(id="mw-content-text")

    links = {}
    prefix = "https://en.wikipedia.org/wiki/"

    for element in body.find_all('a'):
        link = element.get('href')
        title = element.get("title")
        if not link or not title:
            continue
        if (link.startswith(prefix)
                and ":" not in link[len(prefix):]   # skip Template:, File:, Portal: — check title only!
                and "?" not in link
                and "#" not in link):
            links[title] = link

    return links


def _get_closest(link_dictionary, target):
    "Greedy pick whichever article is semantically closest to the target page name, using sentence embedding."

    target_embedding = model.encode(target)

    texts, links = list(link_dictionary.keys()), list(link_dictionary.values())

    embeddings = model.encode(texts)

    scores = util.cos_sim(target_embedding, embeddings)

    max_index = scores.argmax()

    print(scores.max())

    return texts[max_index],links[max_index]
            

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("start", type=str, help="The starting wikipedia page.")
    parser.add_argument("target", type=str, help="The target wikipedia page.")

    args = parser.parse_args()

    current, target = args.start, args.target
    current_link = f"https://en.wikipedia.org/wiki/{current.replace(' ', '_')}" 
    target_link = f"https://en.wikipedia.org/wiki/{target.replace(' ', '_')}"

    attemps = 0

    visited = [current_link]

    start_time = time.time()

    while(current_link != target_link):
        links = _scrape(current_link)
        links = {text : link for text, link in links.items() if link not in visited}  

        if not links:
            print("Hit a dead end.")
            return

        text, link = _get_closest(links, args.target)

        print(f"Visiting {text.replace('_', ' ')} @ {link}")

        current = text
        current_link = link

        visited.append(link)

        attemps += 1

        if attemps > 500: 
            print("Gave up.")
            return

    print(f"Successfully reached {target.replace('_', ' ')}!")
    print(f"It took {attemps} attemps and {time.time() - start_time} seconds.")

if __name__ == "__main__":
    main()