from fastapi import FastAPI
import networkx as nx

app = FastAPI(title="RCA Service")


@app.post("/rca")
def root_cause(events: dict):
    G = nx.DiGraph()
    for e in events["edges"]:
        G.add_edge(e["source"], e["target"])
    causes = [n for n in G.nodes if G.in_degree(n) == 0]
    return {"possible_causes": causes}
