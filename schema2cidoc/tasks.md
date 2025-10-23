# Digital Hermeneutics Cascading Refactor Tasks

## Done
- Create `schema2cidoc/nanopub_generator_utils.py` to build pubInfo first (mint `ex:<doc_id>`), then provenance using consistent FOAF person URIs and `hico:isExtractedFrom ex:<doc_id>`.
- Create `schema2cidoc/cidoc_generator_utils.py` to emit:
  - `ex:facts_<id>`: all entity type declarations (`a crm:*`), `crm:P1_is_identified_by`, appellations with `rdfs:label`, plus established fact events.
  - `ex:assertion_<id>`: authorial argument events only (no entity types/labels).
  - `ex:head_<id>`: nanopub header linking `assertion`, `provenance`, `pubInfo`.
- Create `schema2cidoc/digital_hermeneutics_generator.py` orchestrator to run nanopub + cidoc generation in cascade per document.
- Align pubInfo graph name to `ex:pubInfo_<id>` and header links accordingly.

## To verify
- Provenance `prov:wasAssociatedWith` points to FOAF Person URIs derived from `document_metadata.authors_list` when available; otherwise falls back to `ex:<slug(asserted_by)>` and types them as `foaf:Person`.
- Timestamp format in pubInfo is Zulu: `YYYY-MM-DDTHH:MM:SSZ`.
- No `UnknownSource*` remains; `asserted_by` normalization fallback is `<doc_id>`.
- CIDOC facts graph contains all entity URIs/types/appellations/labels referenced by either facts or assertions.

## Next
- Optional: type `ex:<doc_id>` with bibliographic class (e.g., `fabio:JournalArticle`) in provenance (or in a metadata graph) if desired; currently typed as `prov:Entity` to support provenance linking.
- Optional: add dataset-level index listing `ex:head_<id>` entries across documents.
- Optional: move to `rdflib.Dataset` instead of `ConjunctiveGraph`.

## How to run
```
# From repository root
python schema2cidoc/digital_hermeneutics_generator.py
```

Outputs per document in `schema2cidoc/documents/<id>/`:
- `nanopub.trig` with `ex:pubInfo_<id>` and `ex:provenance_<id>`.
- `cidoc.trig` with `ex:facts_<id>`, `ex:assertion_<id>`, and `ex:head_<id>`.
