from fastapi import FastAPI
import networkx as nx
from typing import List
from pydantic import BaseModel


class Edge(BaseModel):
    source: str
    target: str


class RCAInput(BaseModel):
    edges: List[Edge]


app = FastAPI(title="RCA Service")


@app.post("/rca")
def root_cause(events: RCAInput):
    G = nx.DiGraph()
    for e in events.edges:
        G.add_edge(e.source, e.target)
    causes = [n for n in G.nodes if G.in_degree(n) == 0]
    return {"possible_causes": causes}
