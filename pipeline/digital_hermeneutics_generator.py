# Digital Hermeneutics cascading generator
# Order:
# 1) pubInfo (mint ex:<doc_id>, timestamp, account, software agent, wasDerivedFrom ex:<doc_id>)
# 2) provenance (acts use hico:isExtractedFrom ex:<doc_id> and consistent FOAF agents)
# 3) CIDOC: facts (entity types, appellations, P1 links + established facts)
# 4) CIDOC: assertions (authorial arguments only)

import os
import json
from typing import Any, Dict

from nanopub_generator_utils import generate_nanopub_trig
from cidoc_generator_utils import generate_cidoc_trig


def load_input_index(base_dir: str) -> Dict[str, Any]:
    with open(os.path.join(base_dir, "input.json"), "r", encoding="utf-8") as f:
        return json.load(f)


def load_relations(doc_dir: str) -> Dict[str, Any]:
    rel = os.path.join(doc_dir, "relations.json")
    if os.path.exists(rel):
        with open(rel, "r", encoding="utf-8") as f:
            return json.load(f)
    split = os.path.join(doc_dir, "split_relations.json")
    if os.path.exists(split):
        with open(split, "r", encoding="utf-8") as f:
            return json.load(f)
    return {"work_schema_metadata": {"interpretation_layer": {"nodes": [], "relations": []}}}


def main():
    base = os.path.dirname(__file__)
    docs = os.path.join(base, "documents")
    # Read index from the pipeline folder
    index = load_input_index(base)
    index_by_id = {e.get("file_id"): e for e in index.get("files", [])}

    for doc_id in os.listdir(docs):
        ddir = os.path.join(docs, doc_id)
        if not os.path.isdir(ddir):
            continue
        entry = index_by_id.get(doc_id, {})
        relations_payload = load_relations(ddir)
        # If interpretation.json exists, inject HiCO metadata for provenance
        interp_path = os.path.join(ddir, "interpretation.json")
        if os.path.exists(interp_path):
            try:
                with open(interp_path, "r", encoding="utf-8") as f:
                    interp_data = json.load(f)
                hico_obj = interp_data.get("hico") if isinstance(interp_data, dict) else None
                if isinstance(hico_obj, dict):
                    ws = relations_payload.setdefault("work_schema_metadata", {})
                    il = ws.setdefault("interpretation_layer", {})
                    il["hico"] = hico_obj
            except Exception:
                pass
        # 1+2 nanopub first (as text blocks)
        prov_block, pubinfo_block = generate_nanopub_trig(ddir, doc_id, entry, relations_payload)
        # 3+4 CIDOC (as text blocks)
        facts_graph, assertion_graph = generate_cidoc_trig(ddir, relations_payload, entry, doc_id)
        # Compose single TRIG per document
        out_trig = os.path.join(ddir, "nanopub.trig")
        lines = [
            "@prefix ex: <http://example.org/> .",
            "@prefix np: <http://www.nanopub.org/nschema#> .",
            "@prefix prov: <http://www.w3.org/ns/prov#> .",
            "@prefix dcterms: <http://purl.org/dc/terms/> .",
            "@prefix foaf: <http://xmlns.com/foaf/0.1/> .",
            "@prefix fabio: <http://purl.org/spar/fabio/> .",
            "@prefix frbr: <http://purl.org/vocab/frbr/core#> .",
            "@prefix prism: <http://prismstandard.org/namespaces/basic/2.0/> .",
            "@prefix crm: <http://www.cidoc-crm.org/cidoc-crm/> .",
            "@prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .",
            "@prefix hico: <http://purl.org/emmedi/hico/> .",
            "@prefix cwrc: <http://sparql.cwrc.ca/ontologies/cwrc#> .",
            "@prefix xsd: <http://www.w3.org/2001/XMLSchema#> .",
            "",
            # facts and assertions
            f"ex:facts_{doc_id} {{",
        ]
        if facts_graph:
            lines.append(facts_graph)
        lines.append("}")
        lines.append("")
        lines.append(f"ex:assertion_{doc_id} {{")
        if assertion_graph:
            lines.append(assertion_graph)
        lines.append("}")
        lines.append("")
        # insert provenance and pubInfo blocks captured from rdflib
        if prov_block:
            lines.append(prov_block)
            lines.append("")
        if pubinfo_block:
            lines.append(pubinfo_block)
            lines.append("")
        # Head graph
        lines.append(f"ex:head_{doc_id} {{")
        lines.append(f"    ex:pub_{doc_id} a np:Nanopublication ;")
        lines.append(f"        np:hasAssertion ex:assertion_{doc_id} ;")
        lines.append(f"        np:hasProvenance ex:provenance_{doc_id} ;")
        lines.append(f"        np:hasPublicationInfo ex:pubInfo_{doc_id} .")
        lines.append("}")

        with open(out_trig, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))
        print(f"Wrote {out_trig}")


if __name__ == "__main__":
    main()
