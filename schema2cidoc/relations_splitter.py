# Relations Splitter - separates relations into authorial arguments and established facts
# Reads each document's relations.json and writes two files:
#  - relations_authorial.json (claim_type == "authorial_argument")
#  - relations_factual.json (claim_type == "established_fact")

import json
import os
from typing import Dict, Any, List


def load_json(path: str) -> Dict[str, Any]:
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def save_json(path: str, data: Dict[str, Any]):
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def split_relations(doc_data: Dict[str, Any]) -> (List[Dict[str, Any]], List[Dict[str, Any]]):
    """Return (authorial, factual) relation lists from a relations.json payload."""
    try:
        relations = doc_data["work_schema_metadata"]["interpretation_layer"]["relations"]
    except Exception:
        return [], []

    authorial = [r for r in relations if r.get("claim_type") == "authorial_argument"]
    factual = [r for r in relations if r.get("claim_type") == "established_fact"]
    return authorial, factual


def build_output(original: Dict[str, Any], filtered_relations: List[Dict[str, Any]], label: str) -> Dict[str, Any]:
    """Clone the original payload but replace interpretation_layer.relations and update summary."""
    out = json.loads(json.dumps(original))  # deep copy
    try:
        out["work_schema_metadata"]["interpretation_layer"]["relations"] = filtered_relations
        # update generation summary counts if present
        if "generation_summary" in out["work_schema_metadata"]:
            out["work_schema_metadata"]["generation_summary"]["interpretation_layer_relations"] = len(filtered_relations)
        # annotate metadata
        il_meta = out["work_schema_metadata"]["interpretation_layer"].setdefault("metadata", {})
        il_meta["relation_subset"] = label
    except Exception:
        pass
    return out


def main():
    documents_dir = "./documents"
    os.makedirs(documents_dir, exist_ok=True)

    doc_dirs = [d for d in os.listdir(documents_dir) if os.path.isdir(os.path.join(documents_dir, d))]
    if not doc_dirs:
        print(f"No document directories found in {documents_dir}")
        return

    total = 0
    errors: List[str] = []

    for doc_dir in doc_dirs:
        doc_path = os.path.join(documents_dir, doc_dir)
        rel_file = os.path.join(doc_path, "relations.json")
        if not os.path.exists(rel_file):
            # skip if no relations to split
            continue
        try:
            print(f"Splitting relations in {rel_file}...")
            data = load_json(rel_file)
            authorial, factual = split_relations(data)

            # Build a single combined file that preserves original data and adds split subsets
            combined_out = json.loads(json.dumps(data))  # deep copy
            combined_out["split_relations"] = {
                "authorial_argument": authorial,
                "established_fact": factual
            }

            save_json(os.path.join(doc_path, "split_relations.json"), combined_out)

            print(f"- Wrote split_relations.json with {len(authorial)} authorial and {len(factual)} factual relations for {doc_dir}")
            total += 1
        except Exception as e:
            err = f"Error processing {rel_file}: {e}"
            print(err)
            errors.append(err)

    print(f"\nSplitting complete. {total} files processed.")
    if errors:
        print(f"{len(errors)} errors occurred:")
        for e in errors:
            print(f"- {e}")


if __name__ == "__main__":
    main()
