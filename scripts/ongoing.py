import re
from pathlib import Path

import numpy as np
import pandas as pd
from bs4 import BeautifulSoup
import matplotlib.pyplot as plt

from src.corpus import Loader

case_citation_regex1 = r"<em>([A-Z][\w,.'0-9& ]+)</em> ?v. ?<em>([A-Z][\w,.'0-9& ]+)</em>"
case_citation_regex2 = r"([A-Z][\w,.'0-9& ]+) ?v. ?([A-Z][\w,.'0-9& ]+)"
title_regex = r"([A-Z][\w,.'0-9& ]+) ?v. ?([A-Z][\w,.'0-9& ]+)"
title_exclude_regex = r" \d+ U\.S\. \d+ *"

# docket_regex = r"No. ?([\d-]+)"
# reporting_regex = r"U.?S.|L. ?Ed.( 2d)?|S. ?Ct.|I. ?C. ?C."

def normalise_title(title):
    title = re.sub(r" *,? *$|^ +", "", title)
    title = re.sub(r" +", " ", title)
    return title

def get_title(text, rgx = title_regex):
    title_match = re.search(rgx, text)
    if title_match:
        a, b = title_match.groups()
        b = re.sub(title_exclude_regex, "", b)
        title = f"{a} v. {b}".lower()
        return normalise_title(title.lower())
    else:
        return None

metadata_df = (pd.read_csv("data/caselaw.findlaw.com:opinons/metadata/case_metadata.csv")
  .rename(columns = {"Docket #": "docket_number"}))

file_to_casename = {Loader.url_to_filename(url): casename for url, casename in zip(metadata_df["url"], metadata_df["Description"])}

links = []
folder = Path('data/caselaw.findlaw.com:opinons')
for file in (obj for obj in folder.iterdir() if obj.is_file()):
    with open(file, "r") as f:
        casename = file_to_casename[file.name]
        text = f.read()
        refs = [f"{a} v. {b}" for a, b in re.findall(case_citation_regex1, text) + re.findall(case_citation_regex2, text)]
        for ref in refs:
            links.append((casename, ref))

links_df = (pd.DataFrame(links, columns = ["fro", "to"])
    .assign(weight = 1)
    .groupby(["fro", "to"])
    .agg(weight = ("weight", len))
    .reset_index())

links_df = (links_df.assign(
        fro = [get_title(text, title_regex) for text in links_df["fro"]],
        to = [get_title(text, case_citation_regex2) for text in links_df["to"]]
    ))

metadata_df["to"] = [str(get_title(text, title_regex)) for text in metadata_df["Description"]]

edges_df = (links_df
  .merge(metadata_df[["to"]], on = "to", how = "inner")
  .dropna()
  .drop_duplicates(subset=["fro", "to"])
#   .groupby(["fro", "to"])
#   .agg(weight = ("weight", np.sum))
  .reset_index()
  .query("fro != to"))


import networkx as nx
ntwk = nx.from_pandas_edgelist(edges_df, source = "fro", target = "to", edge_attr = "weight")
comps = nx.connected_components(ntwk)
sizes = [len(x) for x in comps]

ntwk = ntwk.subgraph(max(nx.connected_components(ntwk), key=len))



from graph_tool import Graph, draw
from graph_tool.inference import minimize_blockmodel_dl
import graph_tool.all as gt


nodeset = set(edges_df["fro"]) | set(edges_df["to"])
name2id = {node:idx for idx, node in enumerate(nodeset)}
id2name = {idx:node for idx, node in enumerate(nodeset)}
edgelist = [(name2id[u], name2id[v], w) for u, v, w in zip(edges_df["fro"], edges_df["to"], edges_df["weight"])]
# edgelist = [(u, v, w) for u, v, w in zip(edges_df["fro"], edges_df["to"], edges_df["weight"])]

g = gt.Graph()
eweight = g.new_ep("double")
nodemap = g.add_edge_list(edgelist, hashed=False, eprops=[eweight])
g = gt.GraphView(g, vfilt = gt.label_largest_component(g, directed=False))

nodes = set(g.get_vertices())
trimmed_edgelist = [(a,b,c) for a,b,c in edgelist if a in nodes and b in nodes]
g = gt.Graph()
eweight = g.new_ep("double")
nodemap = g.add_edge_list(edgelist, hashed=False, eprops=[eweight])


clusters = minimize_blockmodel_dl(g)
# Draw the largest component
pos = gt.sfdp_layout(g, eweight=eweight)
clusters.draw(pos=pos, vertex_shape=clusters.get_blocks(),output="clusters.pdf")

# format for networkit pipeline

# nodeset = set(edges_df["fro"]) | set(edges_df["to"])
# name2id = {node:idx for idx, node in enumerate(nodeset)}
# id2name = {idx:node for idx, node in enumerate(nodeset)}

# edgelist_txt = "\n".join([f"({name2id[u]}, {name2id[v]}, {w})" for u,v,w in zip(edges_df["fro"], edges_df["to"], edges_df["weight"])])

# import networkit as nk
# from networkit import vizbridges
# import seaborn
# import ipycytoscape

# graph_reader = nk.graphio.EdgeListReader(separator=",", firstNode=0, continuous=False, directed=True)

# with open("graph.txt", "w") as f:
#     f.write(edgelist_txt)

# ntwk = graph_reader.read("graph.txt")

# vizbridges.widgetFromGraph(ntwk)