import asyncio
import os
import re
from collections import defaultdict
from dataclasses import dataclass
from distutils.command.clean import clean
from enum import Enum, auto

import aiohttp
import pandas as pd
import requests
from bs4 import BeautifulSoup

import config


class FragNote(Enum):
    """Fragrance note can be top note, heart note, or base note.
    Unlabeled notes will be UNKNOWN.
    """

    TOP = auto()
    HEART = auto()
    BASE = auto()
    UNKNOWN = auto()


@dataclass
class FragranceInfo:
    """Simple class for storing basic fragrance info and user reviews.
    """

    brand: str
    name: str
    notes: dict[FragNote, list[str]]
    reviews: list[str]


def get_brand_urls():
    """Get urls to brand pages for the 24 most popular fragrance brands.
    """

    url = "https://basenotes.com/fragrances/"
    page = requests.get(url)
    soup = BeautifulSoup(page.content, "html.parser")

    def is_link_to_popular_brand(url):
        brand_pattern = r".+\?brand=.+[\d]{6}"
        return re.match(brand_pattern, url) is not None

    links = soup.find_all("a")
    popular_brands_tags = [link for link in links if link.has_attr("href")]

    popular_brand_links = [
        link["href"]
        for link in popular_brands_tags
        if is_link_to_popular_brand(link["href"])
    ]

    ordered_frags_prefix = "https://basenotes.com/fragrances/?orderby=popular&"
    return [ordered_frags_prefix + link.split("?")[-1] for link in popular_brand_links]


async def get_brand_frag_urls(brand_url, n_frags: int = 40):
    """Get top n_frags fragrance urls.
    """

    async with aiohttp.ClientSession() as session:

        async with session.get(brand_url) as resp:
            page = await resp.text()

    soup = BeautifulSoup(page, "html.parser")

    def is_link_to_fragrance(url):
        frag_pattern = r".+[\d]{8}"
        return re.match(frag_pattern, url) is not None

    link_tags = soup.find_all(class_="bncard card6")
    frag_url_suffixes = [
        tag.find_all("a")[0]["href"]
        for tag in link_tags
        if is_link_to_fragrance(tag.find_all("a")[0]["href"])
    ]

    frag_url_prefix = "https://basenotes.com"
    return [frag_url_prefix + suffix for suffix in frag_url_suffixes][:n_frags]


async def get_fragrance_info(frag_url, newly_processed):
    """Scrapes the fragrance brand and name; a dictionary of the fragrances
    top notes, heart notes, and base notes; and its user reviews.
    """
    print(f"Starting {frag_url}")
    async with aiohttp.ClientSession() as session:
        async with session.get(frag_url) as resp:
            page = await resp.text()

    soup = BeautifulSoup(page, "html.parser")

    note_tags = soup.find_all("ol")
    try:
        raw_notes = [
            child.text.strip().replace("\t", "").split("\n")
            for child in note_tags[0].children
            if "Notes" in child.text
        ]
        notes = {row[0]: [note for note in row[1:] if note] for row in raw_notes}
        notes_processed = {
            FragNote[note_type.split()[0].upper()]: sorted(notes)
            for note_type, notes in notes.items()
        }
    except IndexError:  # weird errors while scraping, not sure why
        if len(note_tags) > 0:
            notes_processed = {
                FragNote.UNKNOWN: sorted(note_tags[0].get_text().strip().split("\n\n"))
            }
        else:
            notes_processed = {}
    if not any(notes_processed.values()):  # weird errors while scraping, not sure why
        try:
            notes_processed = {
                FragNote.UNKNOWN: sorted(note_tags[0].get_text().strip().split("\n\n"))
            }
        except Exception:
            notes_processed = {}

    name_and_brand = soup.find_all(class_="bnfraginfoheader")
    frag_name = name_and_brand[0].find_all(itemprop="Name")[0].text
    frag_brand = name_and_brand[0].find_all(itemprop="brand")[0].text

    page = requests.get(frag_url + "/reviews")
    soup = BeautifulSoup(page.content, "html.parser")
    review_tags = soup.find_all(class_="reviewtext")
    reviews = [tag.text.strip() for tag in review_tags]

    print(f"Done with {frag_url}")
    newly_processed.append(frag_url)
    return FragranceInfo(frag_brand, frag_name, notes_processed, reviews)


def save_review_data(fragrances: list[FragranceInfo]):
    """Make dataframe and save as CSV, appending if data already exists.
    """
    review_data = defaultdict(list)
    for frag in fragrances:
        for review in frag.reviews:
            review_data["brand"].append(frag.brand)
            review_data["frag"].append(frag.name)
            review_data["review"].append(review)
    df_reviews = pd.DataFrame(review_data)
    df_reviews.to_csv(config.REVIEW_DATA_PATH, index=False, mode="a", header=False)


def save_note_data(fragrances: list[FragranceInfo]):
    """Make dataframe and save as CSV, appending if data already exists.
    """
    note_data = {
        "brand": [frag.brand for frag in fragrances],
        "frag": [frag.name for frag in fragrances],
        "top": [frag.notes.get(FragNote.TOP, "") for frag in fragrances],
        "heart": [frag.notes.get(FragNote.HEART, "") for frag in fragrances],
        "base": [frag.notes.get(FragNote.BASE, "") for frag in fragrances],
        "unknown": [frag.notes.get(FragNote.UNKNOWN, "") for frag in fragrances],
    }
    df_notes = pd.DataFrame(note_data)
    df_notes.to_csv(config.NOTE_DATA_PATH, index=False, mode="a", header=False)


async def get_and_save_brand_data(processed_urls: set[str], brand_url: str):
    """Scrapes fragrance data for a brand and appends to CSV.
    Will skip already scraped fragrances, so can be run multiple times in case
    of error.
    """
    newly_processed = []
    frag_urls = await asyncio.gather(*[get_brand_frag_urls(brand_url, n_frags=40)])
    new_fragrances = await asyncio.gather(
        *[
            get_fragrance_info(frag, newly_processed)
            for brand in frag_urls
            for frag in brand
            if frag not in processed_urls
        ]
    )
    save_note_data(new_fragrances)
    save_review_data(new_fragrances)
    with open("processed_urls.txt", mode="a", encoding="utf-8") as fh:
        for url in newly_processed:
            fh.write(url + "\n")


async def main():
    """Main function.
    """
    if os.path.isfile("processed_urls.txt"):
        with open("processed_urls.txt", mode="r", encoding="utf-8") as fh:
            processed_urls = set(fh.readlines())
    else:
        processed_urls = set()

    for brand_url in get_brand_urls():
        await get_and_save_brand_data(processed_urls, brand_url)

    return


def cleanup():
    """Cleans the reviews a little, as API documentation recommends getting rid of newlines.
    Also gets rid of other excessive white space.
    """
    df = pd.read_csv(
        config.REVIEW_DATA_PATH, header=None, names=["brand", "name", "review"]
    )
    df["cleaned_review"] = df.review.str.replace("\n", " ")
    df.to_csv(config.CLEAN_REVIEW_PATH, index=False)


if __name__ == "__main__":
    asyncio.run(main())
    cleanup()
