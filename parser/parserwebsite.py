import requests
from bs4 import BeautifulSoup as bs
from time import sleep
import xlsxwriter

headers = {'User-Agent': 'CrookedHands/2.0 (EVM x8), CurlyFingers20/1;p'}


def download(url):
    resp = requests.get(url, stream=True)
    r = open(r'D:\Сохранить\Программирование\Python\Парсинг сайтов на Python\image\\' + url.split('/')[-1], 'wb')
    for value in resp.iter_content(1024 * 1024):
        r.write(value)
    r.close()


def get_url():
    for count in range(1, 8):
        url = f"https://scrapingclub.com/exercise/list_basic/?page={count}"
        response = requests.get(url, headers=headers)
        soup = bs(response.text, "lxml")
        data = soup.find_all('div', class_='col-lg-4 col-md-6 mb-4')

        for i in data:
            card_url = 'https://scrapingclub.com' + i.find('a').get('href')
            yield card_url


def array():
    for card_url in get_url():
        response = requests.get(card_url, headers=headers)
        sleep(1)
        soup = bs(response.text, "lxml")
        data = soup.find('div', class_='card mt-4 my-4')
        name = data.find('h3', class_='card-title').text
        price = data.find('h4').text
        text = data.find('p', class_='card-text').text
        url_img = 'https://scrapingclub.com' + data.find('img', class_='card-img-top img-fluid').get('src')
        download(url_img)
        yield name, price, text, url_img


def writer(parametr):
    book = xlsxwriter.Workbook(r'D:\Сохранить\Программирование\Python\Парсинг сайтов на Python\data\data.xlsx')
    page = book.add_worksheet('товар')

    row = 0
    column = 0

    page.set_column('A:A', 20)
    page.set_column('B:B', 20)
    page.set_column('C:C', 50)
    page.set_column('D:D', 50)

    for item in parametr():
        page.write(row, column, item[0])
        page.write(row, column + 1, item[1])
        page.write(row, column + 2, item[2])
        page.write(row, column + 3, item[3])
        row += 1

    book.close()


writer(array)