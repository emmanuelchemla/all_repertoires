"""Static text and labels for the Dash app layout."""

# Each section carries its nav link text/href plus its header texts.
NAV_CONTENT = {
    "brand": "Cross-species Repertoire Explorer & Translator",
    "nav_order": [
        "home",
        "static",
        "translations",
        "one-species",
    ],
    "sections": {
        "home": {
            "link_text": "Home",
            "link_href": "#home",
            "kicker": "Overview",
            "title": "Home",
        },
        "static": {
            "link_text": "Summary plots",
            "link_href": "#static-plots",
            "kicker": "Descriptive plots",
            "title": "Stats for the whole dataset",
        },
        "translations": {
            "link_text": "Single call translation",
            "link_href": "#translations",
            "kicker": "Single call translation",
            "title": "Single call translated to a call in every other species",
        },
        "one-species": {
            "link_text": "Full species repertoire translated",
            "link_href": "#one-species",
            "kicker": "Full species repertoire translated",
            "title": "Mapping every call of a species to a call in every other species",
        },
    },
}

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
