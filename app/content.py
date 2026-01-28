"""Static text and labels for the Dash app layout."""

NAV_LINKS = [
    ("Home", "#home"),
    ("Summary plots", "#static-plots"),
    ("Single species repertoire exploration", "#one-species-explore"),
    ("Single call translated to every other species", "#translations"),
    ("Full species repertoire translated to all species", "#one-species"),
    ("Full repertoire bilingual mapping", "#pair-species"),
]

TODO_ITEMS = [
    "TODO: assess quality of LLM generated database",
    "TODO: review database manually",
    "TODO: make descriptions homogeneous by asking a full rewrite (certainly if we import expert descriptions)",
    "TODO: expand database (more species, more calls)",
    "TODO: provide reviewing/editing tools to the database for external users?",
    "TODO: make the bilingual plot clickable (display the card of a call clicked on, and of its translation)",
    "TODO: fix the issue with automatic scrolling for alignment",
    "TODO: better visualization",
    "TODO: see if text embeddings are doing a good job",
    "TODO: explain 'translation' caveats. One of them: predator does not mean the same thing for all species (compare: president/king)",
]
