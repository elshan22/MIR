from requests import get
from bs4 import BeautifulSoup
from collections import deque
from concurrent.futures import ThreadPoolExecutor, wait
from threading import Lock
import json


class IMDbCrawler:
    headers = {
        'User-Agent': 'Mozilla/5.0 (compatible; MIR_Crawler/1.0)'
    }
    top_250_URL = 'https://www.imdb.com/chart/top'
    base_URL = 'https://www.imdb.com'

    def __init__(self, crawling_threshold=1000):
        self.crawling_threshold = crawling_threshold
        self.not_crawled = deque()
        self.crawled = list()
        self.added_ids = set()
        self.add_list_lock = Lock()
        self.add_queue_lock = Lock()

    def get_id_from_URL(self, URL):
        return URL.split('/')[4]

    def write_to_file_as_json(self):
        with open('IMDB_crawled.json', 'w') as json_file:
            json.dump(list(self.crawled), json_file)

    def read_from_file_as_json(self):
        try:
            with open('IMDB_crawled.json', 'r') as json_file:
                data = json.load(json_file)
                self.crawled = data
                for movie in self.crawled:
                    self.added_ids.add(movie['id'])
        except FileNotFoundError:
            print(f"File 'IMDB_crawled.json' not found. No data loaded.")

    def crawl(self, URL):
        return get(URL, headers=self.headers)

    def extract_top_250(self):
        try:
            top250 = get(self.top_250_URL, headers=self.headers)
            if top250.status_code != 200:
                print('cannot access top 250 movies!')
                return
            soup = BeautifulSoup(top250.content, 'html.parser')
            movie_links = list(map(lambda x: x.a['href'], soup.select('li.ipc-metadata-list-summary-item')))
            for link in movie_links:
                self.not_crawled.append(self.base_URL + link)
                self.added_ids.add(self.get_id_from_URL(self.base_URL + link))
        except:
            print('cannot access top 250 movies!')

    def get_imdb_instance(self):
        return {
            'id': None,  # str
            'title': None,  # str
            'first_page_summary': None,  # str
            'release_year': None,  # str
            'mpaa': None,  # str
            'budget': None,  # str
            'gross_worldwide': None,  # str
            'rating': None,  # str
            'directors': None,  # List[str]
            'writers': None,  # List[str]
            'stars': None,  # List[str]
            'related_links': None,  # List[str]
            'genres': None,  # List[str]
            'languages': None,  # List[str]
            'countries_of_origin': None,  # List[str]
            'summaries': None,  # List[str]
            'synopsis': None,  # List[str]
            'reviews': None,  # List[List[str]]
        }

    def start_crawling(self):
        self.extract_top_250()
        futures = []
        crawled_number = len(self.crawled)

        with ThreadPoolExecutor(max_workers=20) as executor:
            while crawled_number < self.crawling_threshold:
                self.add_queue_lock.acquire()
                URL = self.not_crawled.popleft()
                self.add_queue_lock.release()
                futures.append(executor.submit(self.crawl_page_info, URL))
                crawled_number += 1
                if not len(self.not_crawled):
                    wait(futures)
                    futures = []
            wait(futures)

    def crawl_page_info(self, URL):
        info = self.crawl(URL)
        movie = self.get_imdb_instance()
        self.extract_movie_info(info, movie, URL)
        self.add_list_lock.acquire()
        self.crawled.append(movie)
        self.add_list_lock.release()
        print(f'iteration: {len(self.crawled)}')
        soup = BeautifulSoup(info.content, 'html.parser')
        related_links = list(map(lambda x: x['href'], soup.select('a.ipc-poster-card__title')))
        for link in related_links:
            if self.get_id_from_URL(self.base_URL + link) in self.added_ids: continue
            self.add_list_lock.acquire()
            self.added_ids.add(self.get_id_from_URL(self.base_URL + link))
            self.add_list_lock.release()
            self.add_queue_lock.acquire()
            self.not_crawled.append(self.base_URL + link)
            self.add_queue_lock.release()

    def extract_movie_info(self, res, movie, URL):
        soup = BeautifulSoup(res.content, 'html.parser')
        summary_soup = BeautifulSoup(self.crawl(self.get_summary_link(URL)).content, 'html.parser')
        review_soup = BeautifulSoup(self.crawl(self.get_review_link(URL)).content, 'html.parser')
        movie['id'] = self.get_id_from_URL(URL)
        movie['title'] = self.get_title(soup)
        movie['first_page_summary'] = self.get_first_page_summary(soup)
        movie['release_year'] = self.get_release_year(soup)
        movie['mpaa'] = self.get_mpaa(soup)
        movie['budget'] = self.get_budget(soup)
        movie['gross_worldwide'] = self.get_gross_worldwide(soup)
        movie['directors'] = self.get_director(soup)
        movie['writers'] = self.get_writers(soup)
        movie['stars'] = self.get_stars(soup)
        movie['related_links'] = self.get_related_links(soup)
        movie['genres'] = self.get_genres(soup)
        movie['languages'] = self.get_languages(soup)
        movie['countries_of_origin'] = self.get_countries_of_origin(soup)
        movie['rating'] = self.get_rating(soup)
        movie['summaries'] = self.get_summary(summary_soup)
        movie['synopsis'] = self.get_synopsis(summary_soup)
        movie['reviews'] = self.get_reviews_with_scores(review_soup)

    def get_summary_link(self, url):
        try:
            return '/'.join(url.split('/')[:5]) + '/plotsummary'
        except:
            print("failed to get summary link")

    def get_review_link(self, url):
        try:
            return '/'.join(url.split('/')[:5]) + '/reviews'
        except:
            print("failed to get review link")

    def get_title(self, soup):
        try:
            return soup.select('span.hero__primary-text')[0].text
        except:
            print("failed to get title")

    def get_first_page_summary(self, soup):
        try:
            return soup.find(attrs={'data-testid': 'plot'}).text
        except:
            print("failed to get first page summary")

    def get_director(self, soup):
        try:
            return list(map(lambda x: x.text, soup.select('ul.title-pc-list')[0].find_all('li', recursive=False)[0].ul.find_all('li')))
        except:
            print("failed to get director")

    def get_stars(self, soup):
        try:
            return list(map(lambda x: x.text, soup.select('ul.title-pc-list')[0].find_all('li', recursive=False)[2].ul.find_all('li')))
        except:
            print("failed to get stars")

    def get_writers(self, soup):
        try:
            return list(map(lambda x: x.text, soup.select('ul.title-pc-list')[0].find_all('li', recursive=False)[1].ul.find_all('li')))
        except:
            print("failed to get writers")

    def get_related_links(self, soup):
        try:
            return [self.base_URL + x.a['href'] for x in soup.find(attrs={'data-testid': 'MoreLikeThis'}).select('div.ipc-shoveler')[0].select('div.ipc-sub-grid')[0].find_all('div', recursive=False)]
        except:
            print("failed to get related links")

    def get_summary(self, soup):
        try:
            return [x.div.div.div.text.split('—')[0] for x in soup.find(attrs={'data-testid': 'sub-section-summaries'}).find_all('li')]
        except:
            print("failed to get summary")

    def get_synopsis(self, soup):
        try:
            return [soup.find(attrs={'data-testid': 'sub-section-synopsis'}).text]
        except:
            print("failed to get synopsis")

    def get_reviews_with_scores(self, soup):
        try:
            return [[x.select('div.content')[0].div.text, x.select('span.rating-other-user-rating')[0].span.text if x.select('span.rating-other-user-rating') else None] for x in soup.select('div.lister-list')[0].find_all('div', recursive=False)]
        except:
            print("failed to get reviews")

    def get_genres(self, soup):
        try:
            return [x.text for x in soup.find(attrs={'data-testid': 'genres'}).find_all('a')]
        except:
            print("Failed to get generes")

    def get_rating(self, soup):
        try:
            return soup.find(attrs={'data-testid': 'hero-rating-bar__aggregate-rating__score'}).span.text
        except:
            print("failed to get rating")

    def get_mpaa(self, soup):
        try:
            return soup.select('ul.cdJsTz')[0].find_all('li')[1].text
        except:
            print("failed to get mpaa")

    def get_release_year(self, soup):
        try:
            return soup.select('ul.cdJsTz')[0].find_all('li')[0].text
        except:
            print("failed to get release year")

    def get_languages(self, soup):
        try:
            return [x.text for x in soup.find(attrs={'data-testid': 'title-details-languages'}).find_all('li')]
        except:
            print("failed to get languages")
            return None

    def get_countries_of_origin(self, soup):
        try:
            return [x.text for x in soup.find(attrs={'data-testid': 'title-details-origin'}).find_all('li')]
        except:
            print("failed to get countries of origin")

    def get_budget(self, soup):
        try:
            return soup.find(attrs={'data-testid': 'title-boxoffice-budget'}).div.text.replace('$', '').replace(',', '').split()[0]
        except:
            print("failed to get budget")

    def get_gross_worldwide(self, soup):
        try:
            return soup.find(attrs={'data-testid': 'title-boxoffice-cumulativeworldwidegross'}).div.text.replace('$', '').replace(',', '').split()[0]
        except:
            print("failed to get gross worldwide")


def main():
    imdb_crawler = IMDbCrawler(crawling_threshold=1000)
    imdb_crawler.read_from_file_as_json()
    imdb_crawler.start_crawling()
    imdb_crawler.write_to_file_as_json()


if __name__ == '__main__':
    main()
