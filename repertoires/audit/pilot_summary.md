# Pilot Verification Summary

Generated for the 10-species pilot set:

- `pongo-pygmaeus`
- `cercopithecus-neglectus`
- `globicephala-melas`
- `saccopteryx-bilineata`
- `corvus-brachyrhynchos`
- `leptonychotes-weddellii`
- `forpus-passerinus`
- `hyla-arborea`
- `phyllostomus-discolor`
- `rousettus-aegyptiacus`

## Current Counts

- References checked: 54
- Metadata verified: 43
- Metadata needing review: 11
- Open-access PDFs downloaded: 11
- References still needing manual PDFs or manual access review: 43

## Main Metadata Review Items

- `delgado_2006`: DOI in YAML appears to differ from resolved Crossref DOI.
- `berg_etal_2011`: DOI in YAML appears to differ from resolved Crossref DOI; downloaded author-hosted PDF reports DOI `10.1098/rspb.2011.0932`.
- `schneider_2004`: DOI/metadata resolution does not match the YAML citation and should be manually corrected or replaced.
- `yorzinski_etal_2006`: YAML citation and downloaded PDF support the paper, but the YAML DOI resolves to unrelated metadata and should be corrected.
- `knornschild_2014`: YAML citation says "Vocal production learning in bats"; the YAML DOI resolves to unrelated Frontiers metadata and should be corrected.
- `esch_etal_2009`, `thomas_kuechle_1982`, `courts_etal_2020`, `vester_etal_2014`, `herbert_1983`, and `chamberlain_auger_1990` remain marked for review because automated metadata matching was incomplete or mismatched.

## Primary Inventory Coverage

- `rousettus-aegyptiacus`: primary PDF downloaded and all YAML call terms were found in the primary PDF text.
- `globicephala-melas`: primary PDF downloaded, but only `whistle` matched by simple term search; manual inspection is needed because the paper likely uses broader labels such as stereotypical/variable vocalisations rather than exact YAML names.
- `pongo-pygmaeus`, `saccopteryx-bilineata`, and `leptonychotes-weddellii`: primary PDFs were not downloaded, so call coverage remains manual.
- Species without `primary_inventory.id` have candidate-spine results in `audit/candidate_spine_papers.yaml`.

## Files

- Full manifest: `audit/reference_manifest.yaml`
- Manual PDF queue: `audit/manual_pdf_needed.yaml`
- Candidate spine-paper search results: `audit/candidate_spine_papers.yaml`
- Per-species reports: `audit/species/*.md`
- Downloaded PDFs: `papers/oa/*.pdf`
