import sys
import os

import glob
import json

from bs4 import BeautifulSoup
from matplotlib import pyplot as plt
from ibm_watson import DiscoveryV1
from ibm_cloud_sdk_core.authenticators import IAMAuthenticator
import helper
from helper import fetch_object
http_proxy = ""
https_proxy = ""

http_config = {
    "proxies": {
        "http": http_proxy,
        "https": https_proxy
    }
}


authenticator = IAMAuthenticator(
    apikey=api_key)
authenticator.set_proxies(http_config)

discovery = DiscoveryV1(
    version='2018-12-03',
    authenticator=authenticator
)

discovery.set_service_url(url)
env, env_id = fetch_object(
    discovery, "environment", "Bookworm",
    create=True, create_args=dict(
        description="A space to read and understand stories"))

# print(json.dumps(env.result, indent=2))

# View default configuration
configurations = discovery.list_configurations(environment_id=env_id).result
cfg_id = configurations['configurations'][0]['configuration_id']
print(json.dumps(configurations, indent=2))

data_dir = "/home/houbowei/Artificial_intellectual_disabilities/bookworm/data"
filename = os.path.join(data_dir, "sample.html")

# get collection id
if discovery.list_collections(env_id, name='sample').get_result()['collections'] is None:
    new_collection = discovery.create_collection(
        env_id, name="sample"
    )
col_id = discovery.list_collections(env_id, name='sample').get_result()[
    'collections'][0]['collection_id']
# 如果你已经添加过了document, 先删除, 或者直接获取document的status
with open(filename, "r") as f:
    res = discovery.add_document(
        env_id, col_id, file=f, file_content_type='html').get_result()
print(json.dumps(res, indent=2))

# Prepare a collection of documents to use
col, col_id = helper.fetch_object(discovery, "collection", "Story Chunks", environment_id=env_id,
                                  create=True, create_args=dict(
                                      environment_id=env_id, configuration_id=cfg_id,
                                      description="Stories and plots split up into chunks suitable for answering"
                                  ))
print(json.dumps(col, indent=2))

# Add documents to collection
doc_ids = []  # to store the generated id for each document added
for filename in glob.glob(os.path.join(data_dir, "Star-Wars", "*.html")):
    print("Adding file:", filename)
    with open(filename, "r") as f:
        # Split each individual <p> into its own "document"
        doc = f.read()
        soup = BeautifulSoup(doc, 'html.parser')
        for i, p in enumerate(soup.find_all('p')):
            doc_info = discovery.add_document(
                environment_id=env_id,
                collection_id=col_id,
                file=json.dumps({"text": p.get_text(strip=True)}),
                filename='n',
                file_content_type="application/json",
                metadata=json.dumps({"title": soup.title.get_text(strip=True)})
            )

            doc_ids.append(doc_info.get_result()["document_id"])
print("Total", len(doc_ids), "documents added.")

col, col_id = helper.fetch_object(
    discovery, "collection", "Story Chunks", environment_id=env_id)
print(json.dumps(col, indent=2))
