__author__ = 'Adrian'
from selenium import webdriver
from selenium.webdriver.common.keys import Keys
import sys
import csv
import os
import tokenize as token


def prepare_file(path):
        path = os.path.join(os.path.dirname(__file__), '%s' % path)
        f = open(path)
        tokens = token.generate_tokens(f.readline)
        return tokens


def get_next_node(tokens):
    node_id = tokens.next()[1]
    protein_code = tokens.next()[1]
    return node_id, protein_code

tokens = prepare_file('..\\dataset\\DanioRerio\\danioRerio_layout.txt')
get_next_node(tokens)
tokens.next()
browser = webdriver.Chrome('chromedriver.exe')
while True:
    node_id, protein_code = get_next_node(tokens)
    if node_id in ('13', '89'):
        protein_code = protein_code + tokens.next()[1]
    elif node_id in ('14', '15'):
         protein_code = protein_code + tokens.next()[1] + tokens.next()[1]
    try:
        browser.get("http://refgene.com/")
        search_box = browser.find_element_by_name('q')
        search_box.send_keys(protein_code + Keys.RETURN)
        gene_symbol = browser.find_element_by_xpath('//*[@id="content"]/div/table/tbody/tr[5]/td/a')
        gene_symbol.click()

        i = 2
        while True:

            cell = browser.find_element_by_xpath('//*[@id="content"]/table/tbody/tr['+str(i)+']/td[4]')
            if cell.text == 'Danio rerio':
                break
            else:
                i = i + 1
    except:
        with open('..\\dataset\\DanioRerio\\danioRerio_functions.txt', 'ab') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow([node_id])
        tokens.next()
        continue
    url = browser.find_element_by_xpath('//*[@id="content"]/table/tbody/tr['+str(i)+']/td[1]')
    url.click()

    row = 2
    terms = []
    while True:
        try:
            goterm = browser.find_element_by_xpath('//*[@id="geneontology"]/table/tbody/tr['+str(row)+']/td[1]/a')
            terms.append(goterm.text)
            row = row + 1
        except:
            break
    output = ''
    for term in terms:
        output = output + term + ','
    try:
        if output[output.__len__() - 1] == ',':
            output = output[:-1]
    except:
        pass
    with open('..\\dataset\\DanioRerio\\danioRerio_functions.txt', 'ab') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow([node_id + ',' + output])
    tokens.next()
