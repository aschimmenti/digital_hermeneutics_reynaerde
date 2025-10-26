import os
import json
from typing import Dict, List, Tuple

# Domain-range specification aligned with the relationship_extractor prompt (latest rules)
DOMAIN_RANGE: Dict[str, Tuple[List[str], List[str]]] = {
    # Creation
    "created_by": (["work"], ["person", "organization"]),
    "created_during": (["work"], ["date"]),
    "created_in": (["work"], ["place"]),  # legacy
    "created_at": (["work"], ["place"]),

    # Influence
    "influenced_by": (["work", "person"], ["person", "language", "organization", "historical_context", "concept", "group"]),

    # Spatial/Temporal
    "located_in_space": (["person", "organization", "work"], ["place"]),
    "located_in_time": (["person", "work", "organization"], ["date"]),

    # Membership / Association
    "associated_with": (["person"], ["organization", "group", "person"]),

    # References (STRICT): only from work to these ranges
    "refers_to": (["work"], ["work", "person", "place", "historical_context", "organization"]),

    # Linguistic
    "speaks_language": (["person"], ["language"]),
    "written_in_language": (["work"], ["language"]),

    # Classification
    "has_genre": (["work"], ["genre"]),
    "has_theme": (["work"], ["concept"]),
    "has_characteristic": (["work"], ["concept"]),

    # Biographical
    "has_occupation": (["person"], ["role"]),
    "has_expertise_in": (["person"], ["concept", "language"]),
    "place_of_birth": (["person"], ["place"]),
    "date_of_birth": (["person"], ["date"]),
    "place_of_death": (["person"], ["place"]),
    "date_of_death": (["person"], ["date"]),
    "educated_at": (["person"], ["organization", "place"]),
    "lived_in": (["person"], ["place"]),

    # Roles (scope not enforced here, only basic type check)
    "has_role": (["person"], ["role"]),
}


def validate_relation(rel: Dict[str, str], type_by_id: Dict[str, str]) -> Tuple[bool, str]:
    rtype = rel.get("relation_type")
    s = rel.get("source_id")
    t = rel.get("target_id")
    if rtype not in DOMAIN_RANGE:
        return False, f"unknown_relation_type:{rtype}"
    dom, rng = DOMAIN_RANGE[rtype]
    st = type_by_id.get(s)
    tt = type_by_id.get(t)
    if st is None:
        return False, f"missing_source:{s}"
    if tt is None:
        return False, f"missing_target:{t}"
    if st not in dom:
        return False, f"domain_violation:{rtype}:got={st}:allowed={dom}"
    if tt not in rng:
        return False, f"range_violation:{rtype}:got={tt}:allowed={rng}"
    return True, ""


def load_relations_file(path: str) -> Tuple[List[Dict], Dict[str, str]]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    interp = (data.get("work_schema_metadata") or {}).get("interpretation_layer") or {}
    nodes = interp.get("nodes", [])
    rels = interp.get("relations", [])
    type_by_id = {n.get("id"): n.get("type") for n in nodes if isinstance(n, dict)}
    return rels, type_by_id


def main():
    base = os.path.dirname(__file__)
    docs = os.path.join(base, "documents")
    total = 0
    valid = 0
    by_type = {}
    invalid_samples = []

    for doc_id in sorted(os.listdir(docs)):
        ddir = os.path.join(docs, doc_id)
        if not os.path.isdir(ddir):
            continue
        rel_path = os.path.join(ddir, "relations.json")
        if not os.path.exists(rel_path):
            continue
        try:
            rels, type_by_id = load_relations_file(rel_path)
        except Exception as e:
            print(f"Error reading {rel_path}: {e}")
            continue
        for rel in rels:
            total += 1
            rtype = rel.get("relation_type") or "?"
            ok, reason = validate_relation(rel, type_by_id)
            stats = by_type.setdefault(rtype, {"total": 0, "valid": 0, "invalid": 0})
            stats["total"] += 1
            if ok:
                valid += 1
                stats["valid"] += 1
            else:
                stats["invalid"] += 1
                if len(invalid_samples) < 50:
                    s = rel.get("source_id")
                    t = rel.get("target_id")
                    invalid_samples.append({
                        "doc": doc_id,
                        "relation_type": rtype,
                        "source_id": s,
                        "source_type": type_by_id.get(s),
                        "target_id": t,
                        "target_type": type_by_id.get(t),
                        "reason": reason
                    })

    print("\n=== Relationship Extraction Domain-Range Evaluation ===")
    print(f"Documents scanned: {sum(1 for d in os.listdir(docs) if os.path.isdir(os.path.join(docs, d)))}")
    print(f"Total relations: {total}")
    print(f"Valid relations: {valid} ({(valid/total*100 if total else 0):.1f}%)")
    print(f"Invalid relations: {total - valid} ({(((total - valid)/total*100) if total else 0):.1f}%)\n")

    print("Per relation type:")
    for rtype in sorted(by_type.keys()):
        stats = by_type[rtype]
        tot = stats["total"]
        v = stats["valid"]
        print(f"- {rtype}: {v}/{tot} valid ({(v/tot*100 if tot else 0):.1f}%)")

    if invalid_samples:
        print("\nInvalid samples (up to 50):")
        for s in invalid_samples:
            print(f"- [{s['doc']}] {s['relation_type']}: {s['source_id']} ({s['source_type']}) -> {s['target_id']} ({s['target_type']}) :: {s['reason']}")


if __name__ == "__main__":
    main()
