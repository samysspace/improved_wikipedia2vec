import numpy as np
import os
import requests
import re
import sqlite3
from sqlite3 import OperationalError
import xml.etree.ElementTree as et
import xml.dom.minidom

def strip_tag_name(t):
	"""
	Helper to parse XML
	"""
	t = elem.tag
	idx = k = t.rfind("}")
	if idx != -1:
		t = t[idx + 1:]
	return t

def read_kore_entities(file_name):
	"""
	Helper to read KORE entities for testing.
	"""
	entities = []
	with open(file_name, "r") as kore:
		for entity in kore:
			# Make it Wikipedia-able.
			entity = "_".join(word.split(" "))
			entities.append(entity)
	return entities 

def wiki_top_articles(k=20000):
	"""
	Top k articles according to http://wikirank-2023.di.unimi.it.
	"""
	articles = []
	page_idx = 0

	while len(articles) < k:
		params = {
		    'filter[text]': 'Harmonic centrality',
		    'filter[selected]': 'true',
		    'filter[value]': 'harmonic',
		    'view': 'list',
		    'pageSize': '10',
		    'pageIndex': str(page_idx),
		    'type': 'harmonic',
		    'score': 'false',
		}

		response = requests.get('http://wikirank-2023.di.unimi.it/Q/', params=params, verify=False).json()
		for x in response["items"]:
			idx = x["harmonic"].index("</a>")
			start_idx = x["harmonic"].index(">")
			articles.append(x["harmonic"][start_idx+1:idx])
		page_idx += 1

	return ["_".join(word.split(" ")) for word in articles]

def categories_from_articles(articles, check=False, categories=[]):
	"""
	Given list of articles, output all categories they belong to.

	You will need to download enwiki-latest-categorylinks.sql.gz
	and enwiki-latest-pages-articles.xml.bz2
	from https://dumps.wikimedia.org/enwiki/latest/ and unzip them.
	These are massive files. This command will likely take over a
	day to run depending on hardware but you should only need to
	construct the DB once.
	"""
	conn = sqlite3.connect('wiki.db')
	c = conn.cursor()
	cats = {}
	cats_so_far = set()
	articles_kept = []

	if not os.path.isfile('wiki.db'):
		fd = open('enwiki-latest-categorylinks.sql', 'r')
		sqlFile = fd.read()
		fd.close()

		sqlCommands = sqlFile.split(';')
		for command in sqlCommands:
			try:
				c.execute(command)
			except OperationalError:
				print("Command skipped")

		c.execute("CREATE TABLE page (page_id int, page_title varchar(255));")
		# Have to use iterparse because articles dataset too large
		article_rows = et.iterparse("enwiki-latest-pages-articles.xml", events=('start', 'end'))
		for (event, elem) in article_rows:
			tname = strip_tag_name(elem.tag)
			if event == 'start':
				if tname == 'page':
					title = ''
					id = -1
				elif tname == 'revision':
					revision = True
			else:
				if tname == 'title':
					title = elem.text
				elif tname == 'id' and not revision:
					id = int(elem.text)
				revision = False
			if title != '':
				c.execute(f"INSERT INTO page ({title}, {id})")

	for article in articles:
		try:
			command = f"""
				SELECT cl_to
				FROM categorylinks
				INNER JOIN page ON (cl_from = page_id)
				WHERE page_title={article}
			""".strip()
			result = c.execute(command)
			rows = result.fetchall()
			if check:
				cats[article] = set(rows).intersection(set(categories))
			else:
				cats[article] = rows
			articles_kept.append(article)
			for row in cats[article]:
				cats_so_far.add(row)
				if len(cats_so_far) == 20000:
					break
		except OperationalError:
			print("Command skipped")

	# You will get an assertion error either if you haven't
	# downloaded all the files.
	assert len(cats) == len(articles)
	return articles_kept, cats


def articles_from_categories(categories, 
							 current_article_count,
							 limit=50000, 
							 limit_per_cat=100):
	"""
	Given list of categories, output the articles belonging to these
	categories. We want to provide a limit on total articles to make
	training manageable and a limit per category to make sure that 
	no one category is overwhelming our dataset.
	"""

	# These commands should be fast since we have already set up the DB.
	conn = sqlite3.connect('wiki.db')
	c = conn.cursor()
	running_ct = current_article_count
	articles = []

	for category in categories:
		try:
			command = f"""
				SELECT page_title
				FROM categorylinks
				INNER JOIN page ON (cl_from = page_id)
				WHERE cl_to={category}
				LIMIT {limit_per_cat}
			""".strip()
			result = c.execute(command)
			rows = result.fetchall()
			for row in rows:
				if running_ct != limit:
					articles.append(row)
					running_ct += 1
				else:
					break
		except OperationalError:
			print("Command skipped")

	# Assertion errors occur when files are not imported.
	assert len(articles) == limit - current_article_count
	return articles 

def construct_category_graph(total_articles, categories, train='train'):
	"""
	Given articles list, Construct modified "bag-of-categories" matrix. 
	1 if a category is a direct category of an article, 
	0.8 if a category is a parent of a direct category, and so on and so forth.
	"""
	conn = sqlite3.connect('wiki.db')
	c = conn.cursor()
	data_matrix = [[0 for _ in range(20000)] for _ in range(50000)]

	for idx, article in enumerate(total_articles):
		try:
			command = f"""
				SELECT page_title, a.cl_to, b.cl_to, c.cl_to, d.cl_to, e.cl_to
				FROM page
				INNER JOIN category_links a ON (a.cl_from = page_id)
				INNER JOIN category_links b ON (a.cl_to = b.cl_from)
				INNER JOIN category_links c ON (b.cl_to = c.cl_from)
				INNER JOIN category_links d ON (c.cl_to = d.cl_from)
				INNER JOIN category_links e ON (d.cl_to = e.cl_from)
				WHERE page_title={article}
			""".strip()
			result = c.execute(command)
			rows = result.fetchall()
			for row in rows:
				# We need to take the max since category graphs are cyclical and
				# we want preference on the closest possible ancestor.
				if row[1] in categories:
					data_matrix[idx][categories.index(row[1])] = max(data_matrix[idx][categories.index(row[1])], 1)
				if row[2] in categories:
					data_matrix[idx][categories.index(row[2])] = max(data_matrix[idx][categories.index(row[2])], 0.8)
				if row[3] in categories:
					data_matrix[idx][categories.index(row[3])] = max(data_matrix[idx][categories.index(row[3])], 0.6)
				if row[4] in categories:
					data_matrix[idx][categories.index(row[4])] = max(data_matrix[idx][categories.index(row[4])], 0.4)
				if row[5] in categories:
					data_matrix[idx][categories.index(row[5])] = max(data_matrix[idx][categories.index(row[5])], 0.2)
		except OperationalError:
			print("Command skipped")
	
	# Very sparse matrix. If there is time, I will convert this to sparse representation.
	data_matrix = np.array(data_matrix)
	np.savetxt(f'data_{train}.csv', data_matrix, delimiter=',')
	return data_matrix

def create_xml_bz_cluster_file_for_nn(total_articles, train='train'):
	"""
	Given articles list, construct a xml file consisting of all the articles
	we either wish to train or test over. We then pass this file to our
	modified wikipedia2vec.
	"""
	doc = xml.dom.minidom.Document()

	article_rows = et.iterparse("enwiki-latest-pages-articles.xml", events=('start', 'end'))
	for (event, elem) in article_rows:
		tname = strip_tag_name(elem.tag)
		if event == 'start':
			doc.appendChild(elem)
		else:
			if elem.text in total_articles:
				doc.appendChild(elem)
	fp = open(f'wiki_articles_{train}.xml', 'w')
	doc.writexml(fp, indent='\t', addindent='\t', newl='\n', encoding="utf-8")

	article_file = open('final_article_list.txt', 'w')
	article_file.writelines(total_articles)

def execute_pipeline(train=True):
	"""
	Puts it all together to construct the files and matrices needed for training
	and testing
	"""
	if train:
		top_articles = wiki_top_articles()
	else:
		top_articles = read_kore_entities("kore_entities.txt")

	articles_kept, cats = categories_from_articles(top_articles)
	last_articles = articles_from_categories(list(cats.keys()), len(articles_kept))
	final_article_list = articles_kept + last_articles
	train_word = "train" if train else "test"
	data_matrix = construct_category_graph(final_article_list, cats.keys(), train=train_word)
	create_xml_bz_cluster_file_for_nn(final_article_list, train=train_word)


if __name__ == "__main__":
	execute_pipeline(train=True)

		

