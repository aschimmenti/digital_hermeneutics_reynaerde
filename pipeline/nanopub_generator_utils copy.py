# Utilities to produce nanopublication TRIG (provenance + pubInfo) with cascading coreference
# Ensures consistent URI conventions for persons and document identifiers.

import os
import re
import json
from typing import Any, Dict, List, Set

from rdflib import ConjunctiveGraph, Namespace, URIRef, Literal
from rdflib.namespace import RDF, RDFS, XSD

EX = Namespace("http://example.org/")
PROV = Namespace("http://www.w3.org/ns/prov#")
HICO = Namespace("http://purl.org/emmedi/hico/")
CWRC = Namespace("http://sparql.cwrc.ca/ontologies/cwrc#")
FOAF = Namespace("http://xmlns.com/foaf/0.1/")
FABIO = Namespace("http://purl.org/spar/fabio/")

CERT_SCORES = {
    "possibly": 0.3,
    "possible": 0.3,
    "likely": 0.7,
    "definitively": 1.0,
    "certain": 1.0,
    "uncertain": 0.2,
}


def slug(s: str) -> str:
    s = (s or "").strip().lower()
    s = re.sub(r"[^a-z0-9]+", "_", s)
    s = re.sub(r"_+", "_", s).strip("_")
    return s or "id"


def pascal(s: str) -> str:
    return "".join(p.capitalize() for p in re.split(r"[^A-Za-z0-9]+", s or "") if p)


def normalize_asserted_by(value: str, doc_id: str) -> str:
    if not value:
        return doc_id
    v = str(value).strip().lower()
    if v in {"unknown_source", "unknown", "n/a", "na", "none"}:
        return doc_id
    return value


def score_to_cwrc(score: float) -> URIRef:
    if score >= 0.75:
        return CWRC.high_certainty
    if score >= 0.4:
        return CWRC.medium_certainty
    return CWRC.low_certainty


def person_uri_from_names(family: str, given: str) -> URIRef:
    base = "_".join([p for p in [slug(family), slug(given)] if p]) or "person"
    return EX[base]


def emit_pubinfo_first(cg: ConjunctiveGraph, doc_id: str, model_name: str | None) -> None:
    pub_graph = EX[f"pubInfo_{doc_id}"]
    g = cg.get_context(pub_graph)
    # Mint document node early to allow subsequent coreference
    g.add((EX[doc_id], RDF.type, PROV.Entity))
    # Pub entity and timestamps
    pub = EX[f"pub_{doc_id}"]
    from datetime import datetime, timezone
    g.add((pub, PROV.generatedAtTime, Literal(datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"), datatype=XSD.dateTime)))
    g.add((pub, PROV.wasDerivedFrom, EX[doc_id]))
    g.add((EX.aschimmenti, RDF.type, PROV.Person))
    g.add((EX.aschimmenti, RDFS.label, Literal("aschimmenti")))
    if model_name:
        agent = EX[pascal(model_name) + "Agent"]
        g.add((agent, RDF.type, PROV.SoftwareAgent))
        g.add((agent, RDFS.label, Literal(model_name)))
        # Attribute the publication to the SoftwareAgent
        g.add((pub, PROV.wasAttributedTo, agent))
        # SoftwareAgent acted on behalf of the person
        g.add((agent, PROV.actedOnBehalfOf, EX.aschimmenti))
    else:
        # Fallback: attribute directly to the person
        g.add((pub, PROV.wasAttributedTo, EX.aschimmenti))


def emit_provenance(cg: ConjunctiveGraph, doc_id: str, relations_payload: Dict[str, Any], authors: List[Dict[str, str]] | None = None) -> None:
    prov_graph = EX[f"provenance_{doc_id}"]
    g = cg.get_context(prov_graph)
    # Collect interpretation types/criteria from interpretation.json payload if available
    hico_meta = ((relations_payload.get("work_schema_metadata", {}) or {})
                 .get("interpretation_layer", {}) or {})
    hico_info = (hico_meta.get("hico") or {}) if isinstance(hico_meta.get("hico"), dict) else {}
    # Map provided strings (e.g., "prosopographic_interpretation") to EX terms via Pascal-case
    provided_types = [pascal(t) for t in (hico_info.get("interpretation_type") or []) if isinstance(t, str)]
    provided_criteria = [pascal(c) for c in (hico_info.get("interpretation_criteria") or []) if isinstance(c, str)]
    # Do not declare class nodes here; types belong in facts layer

    # Declare FOAF name properties for authors if given (no rdf:type in provenance)
    for a in (authors or []):
        fam = a.get("family_name", "")
        giv = a.get("given_name", "")
        p = person_uri_from_names(fam, giv)
        if fam:
            g.add((p, FOAF.familyName, Literal(fam)))
        if giv:
            g.add((p, FOAF.givenName, Literal(giv)))

    # Relations
    rels = (relations_payload.get("work_schema_metadata", {}) or {}).get("interpretation_layer", {}).get("relations", [])
    assertion = EX[f"assertion_{doc_id}"]

    # Group scores and methods by asserted_by
    groups: Dict[str, Dict[str, Any]] = {}
    for r in rels:
        if r.get("claim_type") != "authorial_argument":
            continue
        props = (r.get("properties", {}) or {})
        who = normalize_asserted_by(props.get("asserted_by"), doc_id)
        methods = props.get("methods") or props.get("method")
        if isinstance(methods, str):
            methods = [methods]
        methods = [m for m in (methods or [])]
        c_val = CERT_SCORES.get(str(props.get("certainty", "")).lower(), 0.5)
        gentry = groups.setdefault(who, {"scores": [], "methods": set()})
        gentry["scores"].append(c_val)
        for m in methods:
            gentry["methods"].add(pascal(m))

    # Do not emit class declarations for criteria in provenance; keep only links on acts

    # Emit one act per asserted_by group
    for who, agg in groups.items():
        act = EX[f"interpretation_act_{doc_id}_{slug(who)}"]
        g.add((assertion, PROV.wasGeneratedBy, act))
        g.add((act, HICO.isExtractedFrom, EX[doc_id]))
        # Associate with consistent FOAF person URI
        # Try to match against provided authors; fallback to slug(who)
        p_uri = None
        if authors:
            for a in authors:
                fam = (a.get("family_name") or "").strip()
                giv = (a.get("given_name") or "").strip()
                if fam and fam.lower() in who.lower():
                    p_uri = person_uri_from_names(fam, giv)
                    break
        if p_uri is None:
            p_uri = EX[slug(who)]
        g.add((act, PROV.wasAssociatedWith, p_uri))
        avg = sum(agg["scores"]) / len(agg["scores"]) if agg["scores"] else 0.5
        g.add((act, CWRC.hasCertainty, score_to_cwrc(avg)))
        # Attach interpretation types from interpretation.json (all provided)
        for t in provided_types:
            g.add((act, HICO.hasInterpretationType, EX[t]))
        # Attach criteria from interpretation.json and from relation methods
        for c in provided_criteria:
            g.add((act, HICO.hasInterpretationCriterion, EX[c]))
        for c in sorted(agg["methods"]):
            g.add((act, HICO.hasInterpretationCriterion, EX[c]))


def _extract_named_graph_block(trig_text: str, graph_iri: str) -> str:
    # naive extraction: find 'graph {' until matching closing '}' on its own line
    anchor = f"{graph_iri} {{"
    start = trig_text.find(anchor)
    if start == -1:
        return ""
    # find closing '}' after start
    close = trig_text.find("\n}\n", start)
    if close == -1:
        # try end of text
        close = len(trig_text)
    return trig_text[start:close+3].strip()


def generate_nanopub_trig(doc_dir: str, file_id: str, input_entry: Dict[str, Any], relations_payload: Dict[str, Any]) -> tuple[str, str]:
    doc_id = slug(file_id)
    model_name = ((input_entry.get("automated_process_metadata", {}) or {}).get("llm_model_version"))
    authors = (input_entry.get("document_metadata", {}) or {}).get("authors_list") or []

    cg = ConjunctiveGraph()
    # Bind
    cg.bind("ex", EX)
    cg.bind("prov", PROV)
    cg.bind("hico", HICO)
    cg.bind("cwrc", CWRC)
    cg.bind("xsd", XSD)
    cg.bind("rdfs", RDFS)
    cg.bind("foaf", FOAF)
    cg.bind("fabio", FABIO)

    # 1) pubInfo first (mint ex:<doc_id>)
    emit_pubinfo_first(cg, doc_id, model_name)
    # 2) provenance referring to ex:<doc_id> and consistent FOAF persons
    emit_provenance(cg, doc_id, relations_payload, authors=authors)
    # Serialize and extract named graphs as text blocks
    trig_text = cg.serialize(format="trig")
    prov_block = _extract_named_graph_block(trig_text, f"ex:provenance_{doc_id}")
    pubinfo_block = _extract_named_graph_block(trig_text, f"ex:pubInfo_{doc_id}")
    return prov_block, pubinfo_block
