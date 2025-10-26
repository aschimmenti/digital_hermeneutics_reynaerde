# Quiet wrapper API around CidocEventGroupGenerator
# Provides a clean class interface with no stdout noise or file I/O.

from typing import Any, Dict, List
import io
import contextlib

from cidoc_group_generator import CidocEventGroupGenerator


class CidocEventGenerator:
    """Clean, quiet API for generating CIDOC event triples.
    - No prints to stdout
    - No file I/O
    - Returns Turtle fragments (no @prefix unless requested)
    """

    def __init__(self) -> None:
        self._gen = CidocEventGroupGenerator()

    def events_from_payload(self, payload: Dict[str, Any], include_prefixes: bool = False, emit_entities: bool = False) -> str:
        """Generate CIDOC events from a full work-schema payload.
        payload must contain work_schema_metadata.interpretation_layer.nodes/relations
        """
        f = io.StringIO()
        with contextlib.redirect_stdout(f):
            turtle = self._gen.generate_event_rdf_from_data(payload, include_prefixes=include_prefixes, emit_entities=emit_entities)
        return (turtle or "").strip()

    def events_for_claim_type(self, payload: Dict[str, Any], claim_type: str, include_prefixes: bool = False, emit_entities: bool = False) -> str:
        """Generate events filtered by claim_type (e.g., 'established_fact' or 'authorial_argument')."""
        interp = (payload.get("work_schema_metadata", {}) or {}).get("interpretation_layer", {})
        nodes = interp.get("nodes", [])
        relations = [r for r in interp.get("relations", []) if r.get("claim_type") == claim_type]
        ws = {"work_schema_metadata": {"interpretation_layer": {"nodes": nodes, "relations": relations}}}
        return self.events_from_payload(ws, include_prefixes=include_prefixes, emit_entities=emit_entities)

    def events_for_facts(self, payload: Dict[str, Any], include_prefixes: bool = False, emit_entities: bool = False) -> str:
        return self.events_for_claim_type(payload, "established_fact", include_prefixes=include_prefixes, emit_entities=emit_entities)

    def events_for_assertions(self, payload: Dict[str, Any], include_prefixes: bool = False, emit_entities: bool = False) -> str:
        return self.events_for_claim_type(payload, "authorial_argument", include_prefixes=include_prefixes, emit_entities=emit_entities)
