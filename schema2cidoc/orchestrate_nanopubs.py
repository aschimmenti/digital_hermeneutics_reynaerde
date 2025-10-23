# Orchestrator: produce a lightweight per-document bundle TRIG that references
# graphs emitted by nanopub_generator_rdflib.py and cidoc_group_generator.py.
# It does NOT merge or reserialize existing files; instead it writes a manifest
# named graph listing graph IRIs and which file they live in.

import os
import json
from typing import List

PREFIXES = [
    "@prefix ex: <http://example.org/> .",
    "@prefix np: <http://www.nanopub.org/nschema#> .",
    "@prefix prov: <http://www.w3.org/ns/prov#> .",
    "@prefix dcterms: <http://purl.org/dc/terms/> .",
    "@prefix xsd: <http://www.w3.org/2001/XMLSchema#> .",
    "",
]


def make_bundle_trig(doc_id: str, graphs: List[str], files: List[str]) -> str:
    lines = list(PREFIXES)
    bundle_name = f"bundle_{doc_id}"
    lines.append(f"ex:{bundle_name} {{")
    lines.append(f"    ex:{bundle_name} a np:Nanopublication ;")
    # Enumerate known graphs
    for g in graphs:
        # use prov:hasMember to list graphs contained in this bundle
        lines.append(f"        prov:hasMember ex:{g} ;")
    # Attach file hints
    for f in files:
        lines.append(f"        dcterms:hasPart \"{f}\"^^xsd:string ;")
    # finalize
    if lines[-1].endswith(";"):
        lines[-1] = lines[-1][:-1] + "."
    else:
        lines.append("    .")
    lines.append("}")
    return "\n".join(lines)


def main():
    base = os.path.dirname(__file__)
    docs_dir = os.path.join(base, "documents")
    if not os.path.isdir(docs_dir):
        print(f"No documents directory: {docs_dir}")
        return

    for doc_id in os.listdir(docs_dir):
        ddir = os.path.join(docs_dir, doc_id)
        if not os.path.isdir(ddir):
            continue
        nanopub_file = os.path.join(ddir, "nanopub.trig")
        cidoc_file = os.path.join(ddir, "cidoc.trig")
        graphs = []
        files = []
        if os.path.exists(nanopub_file):
            files.append("nanopub.trig")
            graphs.extend([
                f"provenance_{doc_id}",
                f"nanopub_info_{doc_id}",
            ])
        if os.path.exists(cidoc_file):
            files.append("cidoc.trig")
            graphs.extend([
                f"facts_{doc_id}",
                f"assertion_{doc_id}",
                f"head_{doc_id}",
            ])
        if not files:
            continue
        out_path = os.path.join(ddir, "nanopub_bundle.trig")
        with open(out_path, "w", encoding="utf-8") as outf:
            outf.write(make_bundle_trig(doc_id, graphs, files))
        print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
