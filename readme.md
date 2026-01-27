# All repertoires in the world

`database.json` contains an LLM generated list of species with their repertoires

First prompt was:

```text
give me a json with all the calls of a species. The json should contain
- species name
- call name
- acoustic description
- semantic description (context of use)
- scientific references (as many as possible, ideally urls)
- subjective reliability
- comments
- users (only females, only young individuals, only some groups/populations, etc)

Start with chimps
```

`python main.py` generates a dashboard to explore the resulting database
