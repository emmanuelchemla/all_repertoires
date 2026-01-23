# All repertoires in the world

`database.json` contains an LLM generated list of species with their repertoires

`plot.py` generates summary plots

To generate "an interactive translation plot" eg in a notebook use:

```python
from plot import run_interactive

fig = run_interactive(
    json_path="database.json",
    embedding_model="sentence-transformers/all-MiniLM-L6-v2",
    cache=".embedding_cache.json",
    n_components=2,  # Use n_components=3 for a 3D plot
    width=900,
    height=700,
)

fig  # depending on your configuration, also try: fig.show() or run it twice
```
