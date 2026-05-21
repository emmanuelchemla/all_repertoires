# Full-Inventory Subset Notes

For the dataset MVP, we are focusing on the species subset for which we have a conservative, high-confidence full or near-full inventory paper. The goal is not to claim that the rest of the dataset is unusable, but to create a cleaner subset where the repertoire backbone is especially defensible.

Current status: the subset contains 55 species. We have retrieved 36 inventory-paper PDFs so far: 13 automatically with Codex and 23 manually from the HTML table. The remaining 19 selected inventory papers still need verified PDFs.

A full or near-full inventory paper means a source that explicitly enumerates a broad species repertoire, or the main named repertoire classes, across ordinary adult vocal behavior. Canonical broad adult or modality-specific repertoire inventories are acceptable. Narrow studies are excluded: alarm-only papers, song-only papers, food-call papers, courtship-only papers, single-call-family papers, or cases where the rationale says no satisfactory inventory exists. Borderline cases default to excluded.

This subset is marked with the top-level `has_full_inventory` field in each species YAML. Species with `has_full_inventory: true` should have a `primary_inventory.id` pointing to the inventory source used as the high-confidence spine.

We are gathering PDFs for these inventory papers in two passes. First, Codex tries to retrieve the PDFs automatically from DOIs, source URLs, publisher pages, repositories, and open-access mirrors. Second, the remaining sources are listed in `repertoires/inventory_papers/manual_access.html` so they can be downloaded manually where access is available.

The HTML table tracks all selected inventory papers, with retrieval status (`Automatically`, `Manually`, or `No`) and a `Scan` flag for PDFs that are scans or scan-derived OCR copies rather than clean born-digital PDFs. Retrieved PDFs are stored in `repertoires/inventory_papers/` and named as `genus-species__reference_id.pdf`.

When adding PDFs manually, verify that the file matches the selected species and `primary_inventory` citation before committing it. If a downloaded file is a review, a later/earlier paper with a similar title, a chapter from the wrong source, or otherwise not the selected inventory paper, leave it out and keep the species marked as missing.
