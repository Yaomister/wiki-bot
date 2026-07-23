import requests
import argparse
from bs4 import BeautifulSoup
from sentence_transformers import SentenceTransformer, util



headers = {
    "User-Agent": "wiki-bot/1.0"
}

model = SentenceTransformer("all-MiniLM-L6-v2")

def _scrape(current_link):
    html = requests.get(current_link, headers=headers).text
    soup = BeautifulSoup(html, "html.parser")

    body = soup.find(id="mw-content-text")

    links = {}

    for element in body.find_all('a'):
        link = element.get('href')
        text = element.get_text()
        if link and "wikipedia" in link and len(text.strip()) > 0:
            actual_destination = link.split('/')[-1]
            links[text] = f"https://en.wikipedia.org/wiki/{actual_destination}" 


    return links


def _get_closest(link_dictionary, target):
    target = target.replace(' ', '_')

    target_embedding = model.encode(target)

    texts, links = list(link_dictionary.keys()), list(link_dictionary.values())

    embeddings = model.encode(texts)

    scores = util.cos_sim(target_embedding, embeddings)

    max_index = scores.argmax()

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

        if attemps > 50:
            print("Gave up.")
            return

    print(f"Successfully reached {target.replace('_', ' ')}!")
    print(f"It took {attemps} attemps.")



if __name__ == "__main__":
    main()

