# Nanopublication generator using rdflib (Layers 1+2)
# Produces a single TRIG per document combining:
# - Default graph: bibliographic metadata (FABIO/DC/PRISM/FOAF/FRBR)
# - Named graph ex:provenance_<id>: Interpretation context (PROV/HICO/CWRC)
# - Named graph ex:pubInfo_<id>: Publication info (PROV) with attribution/timestamps/derivations and software agent

from datetime import datetime, timezone
import json
import os
import re
from typing import Any, Dict, List, Tuple, Set

from rdflib import ConjunctiveGraph, Graph, Namespace, URIRef, BNode, Literal
from rdflib.namespace import DC, RDF, RDFS, XSD
DC_EDITOR = URIRef("http://purl.org/dc/elements/1.1/editor")

BASE = "http://example.org/"
EX = Namespace(BASE)
FABIO = Namespace("http://purl.org/spar/fabio/")
PRISM = Namespace("http://prismstandard.org/namespaces/basic/2.0/")
FOAF = Namespace("http://xmlns.com/foaf/0.1/")
FRBR = Namespace("http://purl.org/vocab/frbr/core#")
PROV = Namespace("http://www.w3.org/ns/prov#")
HICO = Namespace("http://purl.org/emmedi/hico/")
CWRC = Namespace("http://sparql.cwrc.ca/ontologies/cwrc#")

CERT_SCORES = {
    "possibly": 0.3,
    "possible": 0.3,
    "likely": 0.7,
    "definitively": 1.0,
    "certain": 1.0,
    "uncertain": 0.2,
}

def score_to_cwrc(score: float) -> URIRef:
    if score >= 0.75:
        return CWRC.high_certainty
    if score >= 0.4:
        return CWRC.medium_certainty
    return CWRC.low_certainty


def slug(s: str) -> str:
    s = s.strip().lower()
    s = re.sub(r"[^a-z0-9]+", "_", s)
    s = re.sub(r"_+", "_", s).strip("_")
    return s or "id"


def short_title_slug(title: str, max_tokens: int = 4) -> str:
    if not title:
        return "book"
    tokens = [t.lower() for t in re.split(r"[^A-Za-z0-9]+", title) if t]
    return "_".join(tokens[:max_tokens]) or "book"


def person_id_from_name(family: str, given: str) -> str:
    if family:
        return slug(family)
    if given:
        return slug(given)
    return "person"


def normalize_asserted_by(value: str, doc_id: str) -> str:
    if not value:
        return doc_id
    v = str(value).strip().lower()
    if v in {"unknown_source", "unknown", "n/a", "na", "none"}:
        return doc_id
    return value


def add_person(g: Graph, pid: str, family: str, given: str):
    p = EX[pid]
    g.add((p, RDF.type, FOAF.Person))
    if family:
        g.add((p, FOAF.familyName, Literal(family)))
    if given:
        g.add((p, FOAF.givenName, Literal(given)))


def iter_authors(meta: Dict[str, Any]) -> List[Tuple[str, str, str]]:
    raw = meta.get("authors_list") or meta.get("authors")
    out: List[Tuple[str, str, str]] = []
    for a in raw or []:
        if isinstance(a, dict):
            pid = person_id_from_name(a.get("family_name", ""), a.get("given_name", ""))
            out.append((pid, a.get("family_name", ""), a.get("given_name", "")))
    return out


def emit_metadata_default_graph(g: Graph, file_id: str, meta: Dict[str, Any]):
    doc_id = slug(file_id)
    doc = EX[doc_id]
    doc_type = (meta.get("type") or "").lower()

    # Common fields
    if doc_type == "journal_article":
        g.add((doc, RDF.type, FABIO.JournalArticle))
    elif doc_type == "book_chapter":
        g.add((doc, RDF.type, FABIO.BookChapter))
    else:
        g.add((doc, RDF.type, FABIO.Book))

    title = meta.get("title", "")
    g.add((doc, DC.title, Literal(title)))

    authors = iter_authors(meta)
    if authors:
        for pid, _, _ in authors:
            g.add((doc, DC.creator, EX[pid]))

    date = meta.get("date")
    if date:
        g.add((doc, DC.date, Literal(date)))

    # Journal article specifics
    if doc_type == "journal_article":
        jtitle = meta.get("journal") or meta.get("container", {}).get("title") or ""
        vol = meta.get("volume") or meta.get("container", {}).get("volume") or ""
        vol_id = slug(f"{jtitle}_vol_{vol}") if jtitle and vol else slug(f"{jtitle}_vol") if jtitle else slug("volume")
        vol_node = EX[vol_id]
        g.add((doc, FRBR.partOf, vol_node))
        g.add((vol_node, RDF.type, FABIO.JournalVolume))
        if jtitle:
            g.add((vol_node, DC.title, Literal(jtitle)))
        if vol:
            g.add((vol_node, PRISM.volume, Literal(vol)))
        j_node = EX[slug(f"{jtitle}_journal")] if jtitle else EX[slug("journal")]
        g.add((vol_node, FRBR.partOf, j_node))
        g.add((j_node, RDF.type, FABIO.Journal))
        if jtitle:
            g.add((j_node, DC.title, Literal(jtitle)))
        if meta.get("doi"):
            g.add((doc, PRISM.doi, Literal(meta["doi"])) )

    # Book specifics
    if doc_type == "book":
        if meta.get("publisher"):
            g.add((doc, DC.publisher, Literal(meta["publisher"])))
        if meta.get("doi"):
            g.add((doc, PRISM.doi, Literal(meta["doi"])) )

    # Book chapter specifics
    if doc_type == "book_chapter":
        container = meta.get("container", {}) if isinstance(meta.get("container"), dict) else {}
        book_title = container.get("title", "")
        book_id = short_title_slug(book_title) if book_title else "book"
        book_node = EX[book_id]
        g.add((doc, FRBR.partOf, book_node))
        g.add((book_node, RDF.type, FABIO.Book))
        if book_title:
            g.add((book_node, DC.title, Literal(book_title)))
        # editors
        for e in container.get("editors", []) or []:
            if isinstance(e, dict):
                pid = person_id_from_name(e.get("family_name", ""), e.get("given_name", ""))
                g.add((book_node, DC_EDITOR, EX[pid]))
                add_person(g, pid, e.get("family_name", ""), e.get("given_name", ""))
        publisher = meta.get("publisher") or container.get("publisher")
        if publisher:
            g.add((book_node, DC.publisher, Literal(publisher)))
        series = container.get("series", {}) if isinstance(container.get("series"), dict) else {}
        seq = series.get("volume")
        if seq:
            g.add((book_node, FABIO.hasSequenceIdentifier, Literal(seq)))
        # pages
        sp = meta.get("start_page") or container.get("start_page") or meta.get("prism:startingPage")
        ep = meta.get("end_page") or container.get("end_page") or meta.get("prism:endingPage")
        if sp:
            g.add((doc, PRISM.startingPage, Literal(sp)))
        if ep:
            g.add((doc, PRISM.endingPage, Literal(ep)))
        # backlink
        g.add((book_node, FRBR.part, doc))

    # people blocks for authors
    for pid, fam, giv in authors:
        add_person(g, pid, fam, giv)

    # Declare interpretation vocabulary in default graph
    g.add((EX.PhilologicalAnalysis, RDF.type, HICO.InterpretationType))


def build_provenance_graph(g: Graph, prov_g: Graph, doc_id: str, relations_payload: Dict[str, Any], meta: Dict[str, Any]):
    # Criteria URIs declared inside the provenance named graph
    rels = relations_payload.get("work_schema_metadata", {}).get("interpretation_layer", {}).get("relations", [])
    criteria: Set[str] = set()
    for r in rels:
        if r.get("claim_type") != "authorial_argument":
            continue
        props = r.get("properties", {}) or {}
        method = props.get("method") or props.get("methods")
        methods: List[str] = []
        if isinstance(method, list):
            methods = [str(m) for m in method]
        elif isinstance(method, str):
            methods = [method]
        for m in methods:
            label = "".join(p.capitalize() for p in re.split(r"[^A-Za-z0-9]+", m) if p)
            if label:
                criteria.add(label)
    for c in sorted(criteria):
        prov_g.add((EX[c], RDF.type, HICO.InterpretationCriterion))

    # Authorial groups
    authorial = [r for r in rels if r.get("claim_type") == "authorial_argument"]
    if not authorial:
        return

    # time
    started = meta.get("date")
    year = None
    if isinstance(started, str):
        m = re.search(r"(\d{4})", started)
        if m:
            year = m.group(1)

    # groups by asserted_by
    group: Dict[str, Dict[str, Any]] = {}
    for r in authorial:
        props = r.get("properties", {}) or {}
        who = normalize_asserted_by(props.get("asserted_by"), doc_id)
        method = props.get("method") or props.get("methods")
        methods: List[str] = []
        if isinstance(method, list):
            methods = [str(m) for m in method]
        elif isinstance(method, str):
            methods = [method]
        c_val = CERT_SCORES.get(str(props.get("certainty", "")).lower(), 0.5)
        gentry = group.setdefault(who, {"scores": [], "methods": set()})
        gentry["scores"].append(c_val)
        for m in methods:
            label = "".join(p.capitalize() for p in re.split(r"[^A-Za-z0-9]+", m) if p)
            if label:
                gentry["methods"].add(label)

    assertion = EX[f"assertion_{doc_id}"]
    # Strengthen model: assertion as a prov:Entity (optional but helpful)
    prov_g.add((assertion, RDF.type, PROV.Entity))

    for who, agg in group.items():
        act = EX[f"interpretation_act_{doc_id}_{slug(who)}"]
        prov_g.add((assertion, PROV.wasGeneratedBy, act))
        prov_g.add((act, RDF.type, HICO.InterpretationAct))
        # Use the document URI as the study/publication reference to keep co-reference consistent
        study = EX[doc_id]
        prov_g.add((act, HICO.isExtractedFrom, study))
        # Agent from asserted_by, fallback to document-based agent
        agent = EX[f"{slug(who)}_agent"]
        prov_g.add((act, PROV.wasAssociatedWith, agent))
        avg = sum(agg["scores"]) / len(agg["scores"]) if agg["scores"] else 0.5
        prov_g.add((act, CWRC.hasCertainty, score_to_cwrc(avg)))
        prov_g.add((act, HICO.hasInterpretationType, EX.PhilologicalAnalysis))
        for c in sorted(agg["methods"]):
            prov_g.add((act, HICO.hasInterpretationCriterion, EX[c]))
        if year:
            prov_g.add((act, PROV.startedAtTime, Literal(year, datatype=XSD.gYear)))


def build_pubinfo_graph(pub_g: Graph, doc_id: str, relations_payload: Dict[str, Any], model_name: str):
    account = EX.aschimmenti
    pub = EX[f"pub_{doc_id}"]
    pub_g.add((pub, PROV.wasAttributedTo, account))
    now_iso = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    pub_g.add((pub, PROV.generatedAtTime, Literal(now_iso, datatype=XSD.dateTime)))

    # wasDerivedFrom the publication document itself to ensure co-reference
    rels = relations_payload.get("work_schema_metadata", {}).get("interpretation_layer", {}).get("relations", [])
    # Always derive from the document URI
    pub_g.add((pub, PROV.wasDerivedFrom, EX[doc_id]))

    # account as Person
    pub_g.add((account, RDF.type, PROV.Person))
    pub_g.add((account, RDFS.label, Literal("aschimmenti")))

    # software agent
    if model_name:
        m_id = "".join(p.capitalize() for p in re.split(r"[^A-Za-z0-9]+", model_name) if p) + "Agent"
        agent = EX[m_id]
        pub_g.add((agent, RDF.type, PROV.Agent))
        pub_g.add((agent, RDF.type, PROV.SoftwareAgent))
        pub_g.add((agent, RDFS.label, Literal(model_name)))


def main():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    documents_dir = os.path.join(base_dir, "schema2cidoc", "documents")
    input_path = os.path.join(base_dir, "input.json")

    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    meta_map = {e.get("file_id"): e.get("document_metadata", {}) for e in data.get("files", [])}
    model_map = {e.get("file_id"): (e.get("automated_process_metadata", {}) or {}).get("llm_model_version") for e in data.get("files", [])}

    for file_id, meta in meta_map.items():
        doc_dir = os.path.join(documents_dir, file_id)
        if not os.path.isdir(doc_dir):
            continue
        rel_path = os.path.join(doc_dir, "relations.json")
        if not os.path.exists(rel_path):
            alt = os.path.join(doc_dir, "split_relations.json")
            rel_path = alt if os.path.exists(alt) else None
        relations_payload: Dict[str, Any] = {}
        if rel_path and os.path.exists(rel_path):
            with open(rel_path, "r", encoding="utf-8") as rf:
                relations_payload = json.load(rf)

        # Build a conjunctive graph
        cg = ConjunctiveGraph()
        # Bind prefixes
        cg.bind("ex", EX)
        cg.bind("fabio", FABIO)
        cg.bind("dc", DC)
        cg.bind("prism", PRISM)
        cg.bind("foaf", FOAF)
        cg.bind("frbr", FRBR)
        cg.bind("prov", PROV)
        cg.bind("hico", HICO)
        cg.bind("cwrc", CWRC)
        cg.bind("xsd", XSD)
        cg.bind("rdfs", RDFS)

        # Provenance named graph (no bibliographic metadata here)
        prov_graph_uri = EX[f"provenance_{slug(file_id)}"]
        prov_g = cg.get_context(prov_graph_uri)
        g = cg.default_context  # default graph intentionally left empty
        build_provenance_graph(g, prov_g, slug(file_id), relations_payload, meta)

        # PubInfo named graph (template expects pubInfo_<id>)
        pub_graph_uri = EX[f"pubInfo_{slug(file_id)}"]
        pub_g = cg.get_context(pub_graph_uri)
        build_pubinfo_graph(pub_g, slug(file_id), relations_payload, model_map.get(file_id))

        out_path = os.path.join(doc_dir, "nanopub.trig")
        cg.serialize(destination=out_path, format="trig")
        print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
