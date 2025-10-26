# Utilities to produce CIDOC-CRM TRIG split into facts/assertions
# Reuses CidocEventGroupGenerator with controls to keep entity declarations in facts only.

import json
import os
from typing import Dict, Any, List, Tuple

from cidoc_group_generator import CidocEventGroupGenerator
from cidoc_event_generator_rdflib import CidocEventGeneratorRDFlib
from rdflib import ConjunctiveGraph, Namespace, Literal
from rdflib.namespace import RDF, RDFS

EX = Namespace("http://example.org/")
DCTERMS = Namespace("http://purl.org/dc/terms/")
FOAF = Namespace("http://xmlns.com/foaf/0.1/")
FABIO = Namespace("http://purl.org/spar/fabio/")
FRBR = Namespace("http://purl.org/vocab/frbr/core#")
PRISM = Namespace("http://prismstandard.org/namespaces/basic/2.0/")

def _slug(s: str) -> str:
    import re
    s = (s or "").strip().lower()
    s = re.sub(r"[^a-z0-9]+", "_", s)
    s = re.sub(r"_+", "_", s).strip("_")
    return s or "id"


def _person_id(family: str, given: str) -> str:
    parts = [p for p in [_slug(family), _slug(given)] if p]
    return parts[0] if parts else "person"


def build_biblio_facts(facts_g, doc_id: str, input_entry: Dict[str, Any]) -> None:
    dm = (input_entry.get("document_metadata", {}) or {})
    doc_type = (dm.get("type") or "").lower()
    doc = EX[doc_id]
    # Type
    if doc_type == "journal_article":
        facts_g.add((doc, RDF.type, FABIO.JournalArticle))
    elif doc_type == "book_chapter":
        facts_g.add((doc, RDF.type, FABIO.BookChapter))
    elif doc_type == "book":
        facts_g.add((doc, RDF.type, FABIO.Book))
    else:
        facts_g.add((doc, RDF.type, FABIO.JournalArticle))
    # Title and date
    title = dm.get("title")
    if title:
        facts_g.add((doc, DCTERMS.title, Literal(title)))
    date = dm.get("date")
    if date:
        facts_g.add((doc, DCTERMS.date, Literal(date)))
    # Authors
    authors = dm.get("authors_list") or []
    creator_ids: List[str] = []
    for a in authors:
        if isinstance(a, dict):
            pid = _person_id(a.get("family_name", ""), a.get("given_name", ""))
            creator_ids.append(pid)
            p = EX[pid]
            facts_g.add((p, RDF.type, FOAF.Person))
            fam = a.get("family_name"); giv = a.get("given_name")
            if fam:
                facts_g.add((p, FOAF.familyName, Literal(fam)))
            if giv:
                facts_g.add((p, FOAF.givenName, Literal(giv)))
    for pid in creator_ids:
        facts_g.add((doc, DCTERMS.creator, EX[pid]))
    # Container
    if doc_type == "journal_article":
        jtitle = dm.get("journal") or (dm.get("container", {}) or {}).get("title")
        vol = dm.get("volume") or (dm.get("container", {}) or {}).get("volume")
        vol_id = _slug(f"{jtitle}_vol_{vol}") if jtitle and vol else _slug(f"{jtitle}_vol") if jtitle else _slug("volume")
        vol_node = EX[vol_id]
        facts_g.add((doc, FRBR.partOf, vol_node))
        facts_g.add((vol_node, RDF.type, FABIO.JournalVolume))
        if jtitle:
            facts_g.add((vol_node, DCTERMS.title, Literal(jtitle)))
        if vol:
            facts_g.add((vol_node, PRISM.volume, Literal(vol)))
        jnode = EX[_slug(f"{jtitle}_journal") if jtitle else _slug("journal")]
        facts_g.add((vol_node, FRBR.partOf, jnode))
        facts_g.add((jnode, RDF.type, FABIO.Journal))
        if jtitle:
            facts_g.add((jnode, DCTERMS.title, Literal(jtitle)))
    elif doc_type == "book_chapter":
        container = (dm.get("container") or {}) if isinstance(dm.get("container"), dict) else {}
        btitle = container.get("title") or ""
        book_id = _slug(btitle) if btitle else "book"
        book_node = EX[book_id]
        facts_g.add((doc, FRBR.partOf, book_node))
        facts_g.add((book_node, RDF.type, FABIO.Book))
        if btitle:
            facts_g.add((book_node, DCTERMS.title, Literal(btitle)))


def _extract_named_graph_block(trig_text: str, graph_iri: str) -> str:
    anchor = f"{graph_iri} {{"
    start = trig_text.find(anchor)
    if start == -1:
        return ""
    close = trig_text.find("\n}\n", start)
    if close == -1:
        close = len(trig_text)
    return trig_text[start:close+3].strip()


def generate_cidoc_trig(doc_dir: str, relations_payload: Dict[str, Any], input_entry: Dict[str, Any], file_id: str) -> str:
    doc_id = _slug(file_id)
    interp = (relations_payload.get("work_schema_metadata", {}) or {}).get("interpretation_layer", {})
    nodes = interp.get("nodes", [])
    relations = interp.get("relations", [])

    facts = [r for r in relations if r.get("claim_type") == "established_fact"]
    assertions = [r for r in relations if r.get("claim_type") == "authorial_argument"]

    base_ws = {"work_schema_metadata": {"interpretation_layer": {"nodes": nodes, "relations": []}}}
    combined_ws = json.loads(json.dumps(base_ws))
    combined_ws["work_schema_metadata"]["interpretation_layer"]["relations"] = relations

    # Entity block (facts layer) across all relations
    gen_entities = CidocEventGroupGenerator()
    entity_block = gen_entities.generate_grouped_rdf_content(
        {e["id"]: e for e in nodes},
        gen_entities.group_relations_by_event(combined_ws["work_schema_metadata"]["interpretation_layer"]["relations"]),
        include_prefixes=False,
        emit_entities=True,
    )
    entity_block_str = "\n".join(entity_block).strip()

    # Facts: event triples will be emitted directly into rdflib below

    # Assertions: event triples will be emitted directly into rdflib below

    # Build via rdflib into named graphs and return serialized blocks
    cg = ConjunctiveGraph()
    # Bind prefixes for parsing Turtle into the named graphs
    cg.bind("ex", EX)
    cg.bind("crm", Namespace("http://www.cidoc-crm.org/cidoc-crm/"))
    cg.bind("rdfs", RDFS)
    cg.bind("dcterms", DCTERMS)
    cg.bind("foaf", FOAF)
    cg.bind("fabio", FABIO)
    cg.bind("frbr", FRBR)
    cg.bind("prism", PRISM)
    cg.bind("np", Namespace("http://www.nanopub.org/nschema#"))

    facts_name = EX[f"facts_{doc_id}"]
    assert_name = EX[f"assertion_{doc_id}"]
    facts_g = cg.get_context(facts_name)
    assert_g = cg.get_context(assert_name)

    # Add bibliographic facts via rdflib directly
    build_biblio_facts(facts_g, doc_id, input_entry)

    # Prepare a Turtle header for parsing entity block only
    header = "\n".join([
        "@prefix ex: <http://example.org/> .",
        "@prefix crm: <http://www.cidoc-crm.org/cidoc-crm/> .",
        "@prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .",
        "@prefix dcterms: <http://purl.org/dc/terms/> .",
        "@prefix foaf: <http://xmlns.com/foaf/0.1/> .",
        "@prefix fabio: <http://purl.org/spar/fabio/> .",
        "@prefix frbr: <http://purl.org/vocab/frbr/core#> .",
        "@prefix prism: <http://prismstandard.org/namespaces/basic/2.0/> .",
        "",
    ])
    if entity_block_str:
        facts_g.parse(data=header + entity_block_str, format="turtle")
    # Emit event triples directly via rdflib wrapper
    eg = CidocEventGeneratorRDFlib()
    eg.emit_facts_events(cg, doc_id, relations_payload)
    eg.emit_assertion_events(cg, doc_id, relations_payload)

    trig_text = cg.serialize(format="trig")
    facts_block = _extract_named_graph_block(trig_text, f"ex:facts_{doc_id}")
    assertion_block = _extract_named_graph_block(trig_text, f"ex:assertion_{doc_id}")
    return facts_block[len(f"ex:facts_{doc_id} {{\n"): -2] if facts_block else "", assertion_block[len(f"ex:assertion_{doc_id} {{\n"): -2] if assertion_block else ""


def write_cidoc_trig_file(doc_dir: str, doc_id: str, facts_graph: str, assertion_graph: str):
    out_trig = os.path.join(doc_dir, "cidoc.trig")
    lines = [
        "@prefix crm: <http://www.cidoc-crm.org/cidoc-crm/> .",
        "@prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .",
        "@prefix ex: <http://example.org/> .",
        "@prefix np: <http://www.nanopub.org/nschema#> .",
        "@prefix dc: <http://purl.org/dc/elements/1.1/> .",
        "@prefix foaf: <http://xmlns.com/foaf/0.1/> .",
        "@prefix fabio: <http://purl.org/spar/fabio/> .",
        "@prefix frbr: <http://purl.org/vocab/frbr/core#> .",
        "@prefix prism: <http://prismstandard.org/namespaces/basic/2.0/> .",
        "",
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
    lines.append(f"ex:head_{doc_id} {{")
    lines.append(f"    ex:pub_{doc_id} a np:Nanopublication ;")
    lines.append(f"        np:hasAssertion ex:assertion_{doc_id} ;")
    lines.append(f"        np:hasProvenance ex:provenance_{doc_id} ;")
    lines.append(f"        np:hasPublicationInfo ex:pubInfo_{doc_id} .")
    lines.append("}")

    with open(out_trig, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    return out_trig
