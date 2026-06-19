"""Static text and labels for the Dash app layout."""

# Each section carries its nav link text/href plus its header texts.
NAV_CONTENT = {
    "brand": "Cross-species Repertoire Explorer & Translator",
    "nav_order": [
        "home",
        "quick-look",
        "auto-translation",
        "multispecies",
        "mantel",
        "static",
        "conclusion",
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
        "quick-look": {
            "link_text": "One Species",
            "link_href": "#quick-look",
            "kicker": "What the database looks like",
            "title": "One Species: exploring the database",
        },
        "auto-translation": {
            "link_text": "Two Species",
            "link_href": "#auto-translation",
            "kicker": "Building a 'bilingual translation'",
            "title": "Two Species: automatically building a 'bilingual mapping'",
        },
        "mantel": {
            "link_text": "Sounds and Meanings",
            "link_href": "#mantel",
            "kicker": "Do sound-alikes mean the same?",
            "title": "Calls that sound the same are most likely to be semantically similar",
        },
        "multispecies": {
            "link_text": "Many Species",
            "link_href": "#multispecies",
            "kicker": "Building a 'multi-lingual translation'",
            "title": "Many Species: visualizing all-species translation",
        },
        "static": {
            "link_text": "Database counts",
            "link_href": "#static-plots",
            "kicker": "Database counts",
            "title": "Database counts: Who talks about what",
        },
        "conclusion": {
            "link_text": "Conclusions",
            "link_href": "#conclusion",
            "kicker": "Wrap-up",
            "title": "Conclusions",
        },
        "translations": {
            "link_text": "Appendix 1",
            "link_href": "#translations",
            "kicker": "Appendix 1",
            "title": "Appendix 1: Single call translated to a call in every other species",
        },
        "one-species": {
            "link_text": "Appendix 2",
            "link_href": "#one-species",
            "kicker": "Appendix 2",
            "title": "Appendix 2: Mapping every call of a species to a call in every other species",
        },
    },
}

TODO_ITEMS = [
    "TODO: assess quality of LLM generated database",
    "TODO: review database manually",
    "TODO: provide reviewing/editing tools to the database for external users?",
    "TODO: make descriptions homogeneous by asking a full rewrite (certainly if we import expert descriptions)",
    "TODO: expand database (more species, more calls)",
    "TODO: evaluate quality of distance in text embedding space",
]
