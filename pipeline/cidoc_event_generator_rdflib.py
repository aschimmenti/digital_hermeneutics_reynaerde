# RDFlib-backed CIDOC event generator wrapper
# Bridges existing CidocEventGroupGenerator (string/Turtle) into rdflib named graphs.
# Step 1 of refactor: keep grouping/pattern logic, but assemble graphs with rdflib.

from typing import Any, Dict

from rdflib import ConjunctiveGraph, Namespace
from rdflib.namespace import RDFS

from cidoc_group_generator import CidocEventGroupGenerator

EX = Namespace("http://example.org/")
CRM = Namespace("http://www.cidoc-crm.org/cidoc-crm/")

HEADER = "\n".join([
    "@prefix ex: <http://example.org/> .",
    "@prefix crm: <http://www.cidoc-crm.org/cidoc-crm/> .",
    "@prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .",
    "",
])


class CidocEventGeneratorRDFlib:
    def __init__(self) -> None:
        self._bridge = CidocEventGroupGenerator()

    def _generate_events_turtle(self, nodes, relations, emit_entities: bool = False) -> str:
        ws = {"work_schema_metadata": {"interpretation_layer": {"nodes": nodes, "relations": relations}}}
        # Use the bridge to generate just the event relations in Turtle form
        turtle = self._bridge.generate_event_rdf_from_data(ws, include_prefixes=False, emit_entities=emit_entities)
        return turtle.strip()

    def emit_to_named_graph(self, cg: ConjunctiveGraph, graph_name: str, nodes, relations) -> None:
        """Build events for given nodes/relations into the cg named graph."""
        g = cg.get_context(EX[graph_name]) if not graph_name.startswith("http") else cg.get_context(graph_name)
        # Generate events in Turtle and parse into the graph
        turtle = self._generate_events_turtle(nodes, relations, emit_entities=False)
        if turtle:
            g.parse(data=HEADER + turtle, format="turtle")

    def emit_facts_events(self, cg: ConjunctiveGraph, doc_id: str, payload: Dict[str, Any]) -> None:
        interp = (payload.get("work_schema_metadata", {}) or {}).get("interpretation_layer", {})
        nodes = interp.get("nodes", [])
        relations = [r for r in interp.get("relations", []) if r.get("claim_type") == "established_fact"]
        self.emit_to_named_graph(cg, f"facts_{doc_id}", nodes, relations)

    def emit_assertion_events(self, cg: ConjunctiveGraph, doc_id: str, payload: Dict[str, Any]) -> None:
        interp = (payload.get("work_schema_metadata", {}) or {}).get("interpretation_layer", {})
        nodes = interp.get("nodes", [])
        relations = [r for r in interp.get("relations", []) if r.get("claim_type") == "authorial_argument"]
        self.emit_to_named_graph(cg, f"assertion_{doc_id}", nodes, relations)
