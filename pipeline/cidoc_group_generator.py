# CIDOC-CRM Event-Centric Group Generator
# Groups relations by event type and mints complete CIDOC-CRM events with all participants

import json
import os
from typing import Dict, List, Any, Set, Tuple
from collections import defaultdict

# Import pattern functions from cidoc_patterns.py
from cidoc_patterns import (
    pattern_spatial_location, pattern_temporal_location, pattern_type_assignment,
    pattern_influence, pattern_membership, pattern_association,
    pattern_birth_event, pattern_death_event, pattern_activity_participation,
    pattern_event_place, pattern_event_time,
    pattern_expertise, pattern_language_speaker
)

class CidocEventGroupGenerator:
    """
    Event-centric CIDOC-CRM RDF generator that groups relations by event type
    and mints complete events with all participating entities.
    """
    
    def __init__(self):
        # Namespaces
        self.crm = "crm:"
        self.ex = "ex:"
        
        # Output content
        self.rdf_content = []
        
        # Track processed entities and events
        self.processed_entities = set()
        self.processed_events = set()
        
        # Event type mappings - relations that contribute to the same event
        self.event_clusters = {
            "creation": ["created_by", "created_during", "created_in"],
            "association": ["associated_with"],
            "location": ["located_in_space", "located_in_time"],
            "influence": ["influenced_by"],
            "person_language": ["speaks_language"],
            "person_expertise": ["has_expertise_in"],
            "person_occupation": ["has_occupation"],
            "person_role": ["has_role"],
            "person_education": ["educated_at"],
            "person_birth": ["place_of_birth", "date_of_birth"],
            "person_death": ["place_of_death", "date_of_death"],
            "person_residence": ["lived_in"]
        }
        # Include new 'refers_to' relation with association-like grouping
        self.event_clusters["association"].append("refers_to")
        
        # CIDOC-CRM class mappings
        self.entity_mappings = {
            "work": "E89_Propositional_Object",
            "person": "E21_Person", 
            "place": "E53_Place",
            "organization": "E74_Group", 
            "concept": "E55_Type",
            "language": "E56_Language",
            "date": "E52_Time-Span",
            "historical_context": "E4_Period"
        }

    def load_work_schema(self, file_path: str) -> Dict[str, Any]:
        """Load work schema JSON file."""
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)

    def urify_name(self, name: str) -> str:
        """Convert name to URI-safe format."""
        import re
        s = str(name or "")
        # Replace any run of non-alphanumeric characters with underscore
        s = re.sub(r"[^A-Za-z0-9]+", "_", s)
        # Collapse multiple underscores and trim
        s = re.sub(r"_+", "_", s).strip("_")
        return s or "id"

    def group_relations_by_event(self, relations: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
        """Group relations by their target event type and central entity."""
        event_groups = defaultdict(list)
        
        for relation in relations:
            relation_type = relation["relation_type"]
            source_id = relation["source_id"]
            
            # Determine event type and create group key
            event_type = self.get_event_type_for_relation(relation_type)
            if event_type:
                # Group by event type and central entity (usually the source)
                group_key = f"{event_type}_{source_id}"
                event_groups[group_key].append(relation)
        
        return dict(event_groups)

    def get_event_type_for_relation(self, relation_type: str) -> str:
        """Determine which event type a relation belongs to."""
        for event_type, relation_types in self.event_clusters.items():
            if relation_type in relation_types:
                return event_type
        return None

    def collect_entities_for_event_group(self, event_group: List[Dict[str, Any]], 
                                       entities_lookup: Dict[str, Any]) -> Dict[str, Any]:
        """Collect all entities involved in an event group."""
        involved_entities = {}
        
        print(f"\nDEBUG: Collecting entities for event group with {len(event_group)} relations")
        print(f"DEBUG: Available entities in lookup: {len(entities_lookup)}")
        print(f"DEBUG: Entity IDs in lookup: {list(entities_lookup.keys())[:5]}...")
        
        for relation in event_group:
            source_id = relation["source_id"]
            target_id = relation["target_id"]
            
            print(f"DEBUG: Looking for source_id={source_id}, target_id={target_id}")
            
            if source_id in entities_lookup:
                involved_entities[source_id] = entities_lookup[source_id]
                print(f"DEBUG: Found source entity: {source_id}")
            else:
                print(f"DEBUG: Source entity not found: {source_id}")
                
            if target_id in entities_lookup:
                involved_entities[target_id] = entities_lookup[target_id]
                print(f"DEBUG: Found target entity: {target_id}")
            else:
                print(f"DEBUG: Target entity not found: {target_id}")
        
        print(f"DEBUG: Collected {len(involved_entities)} entities")
        return involved_entities

    def add_entity_rdf(self, entity: Dict[str, Any]):
        """Add entity RDF if not already processed."""
        entity_id = entity["id"]
        if entity_id in self.processed_entities:
            return
            
        entity_name = entity["name"]
        entity_type = entity["type"]
        urified_name = self.urify_name(entity_name)
        
        # Get CIDOC-CRM class
        cidoc_class = self.entity_mappings.get(entity_type, "E1_CRM_Entity")
        
        # Add entity
        self.rdf_content.append(f"{self.ex}{urified_name} a {self.crm}{cidoc_class} ;")
        
        # Add appellation for entities that need them (all except dates and concepts)
        if entity_type in ["work", "person", "place", "organization"]:
            appellation_id = f"{urified_name}_appellation"
            self.rdf_content.append(f"    {self.crm}P1_is_identified_by {self.ex}{appellation_id} .")
            self.rdf_content.append("")
            self.rdf_content.append(f"{self.ex}{appellation_id} a {self.crm}E41_Appellation ;")
            self.rdf_content.append(f'    rdfs:label "{entity_name}" .')
        else:
            self.rdf_content.append(f'    rdfs:label "{entity_name}" .')
        
        self.rdf_content.append("")
        self.processed_entities.add(entity_id)

    def collect_creation_influences(self, creation_id: str, entities_lookup: Dict[str, Any]) -> List[str]:
        """Collect all influences that should be attached to this creation event."""
        influences = []
        
        # Find work influences from processed influence relations
        for content_line in self.rdf_content:
            if f"{creation_id} {self.crm}P15_was_influenced_by" in content_line:
                influences.append(content_line.strip())
        
        # Find creator influences that should be redirected to creation
        creator_name = None
        for entity_id, entity in entities_lookup.items():
            if entity["type"] == "person":
                person_name = self.urify_name(entity["name"])
                for content_line in self.rdf_content:
                    if f"{self.ex}{person_name} {self.crm}P15_was_influenced_by" in content_line:
                        # Redirect person influence to creation
                        influence_target = content_line.split(f"{self.crm}P15_was_influenced_by ")[1].rstrip(" .")
                        influences.append(f"{self.ex}{creation_id} {self.crm}P15_was_influenced_by {influence_target} .")
        
        return influences
    
    def mint_creation_event_only(self, group_key: str, event_group: List[Dict[str, Any]], 
                               entities: Dict[str, Any]):
        """Mint a Creation event with all properties consolidated (entities already added)."""
        if group_key in self.processed_events:
            return
            
        # Extract central work entity (usually the source of created_by)
        work_entity = None
        creator_entity = None
        time_entity = None
        place_entity = None
        
        for relation in event_group:
            rel_type = relation["relation_type"]
            source_id = relation["source_id"]
            target_id = relation["target_id"]
            
            if rel_type == "created_by":
                work_entity = entities.get(source_id)
                creator_entity = entities.get(target_id)
            elif rel_type == "created_during":
                time_entity = entities.get(target_id)
            elif rel_type == "created_in":
                place_entity = entities.get(target_id)
        
        if not work_entity or not creator_entity:
            return
        
        # Create the Creation event with all properties in one block
        work_name = self.urify_name(work_entity["name"])
        creation_id = f"{work_name}_Creation"
        
        # Start creation event
        properties = []
        properties.append(f"    {self.crm}P14_carried_out_by {self.ex}{self.urify_name(creator_entity['name'])}")
        properties.append(f"    {self.crm}P94_has_created {self.ex}{work_name}")
        
        # Add time-span if available
        if time_entity:
            properties.append(f"    {self.crm}P4_has_time-span {self.ex}{self.urify_name(time_entity['name'])}")
        
        # Add place if available
        if place_entity:
            properties.append(f"    {self.crm}P7_took_place_at {self.ex}{self.urify_name(place_entity['name'])}")
        
        # Store creation info for influence processing
        self.current_creation_id = creation_id
        self.current_creator_name = self.urify_name(creator_entity["name"])
        
        # Add creation event with properties
        self.rdf_content.append(f"{self.ex}{creation_id} a {self.crm}E65_Creation ;")
        for i, prop in enumerate(properties):
            if i < len(properties) - 1:
                self.rdf_content.append(f"{prop} ;")
            else:
                self.rdf_content.append(f"{prop} .")
        
        self.rdf_content.append("")
        self.processed_events.add(group_key)

    def mint_creation_event(self, group_key: str, event_group: List[Dict[str, Any]], 
                          entities: Dict[str, Any]):
        """Mint a Creation event with all its participants."""
        # Add all involved entities first
        for entity in entities.values():
            self.add_entity_rdf(entity)
        
        # Then create the event
        self.mint_creation_event_only(group_key, event_group, entities)
    
    def correct_influence_polarity(self, relation: Dict[str, Any], entities: Dict[str, Any]) -> Tuple[str, str]:
        """Correct influence relation polarity based on domain and range rules."""
        source_entity = entities.get(relation["source_id"])
        target_entity = entities.get(relation["target_id"])
        
        if not source_entity or not target_entity:
            return None, None
            
        source_type = source_entity["type"]
        target_type = target_entity["type"]
        
        # Domain: [work, person] → Range: [person, organization, concept, place, historical_context]
        # Check if polarity is correct
        valid_domains = ["work", "person"]
        valid_ranges = ["person", "organization", "concept", "place", "historical_context"]
        
        if source_type in valid_domains and target_type in valid_ranges:
            # Correct polarity
            return relation["source_id"], relation["target_id"]
        elif target_type in valid_domains and source_type in valid_ranges:
            # Reversed polarity - swap
            return relation["target_id"], relation["source_id"]
        else:
            # Invalid relation - skip
            return None, None
    
    def find_creation_event_for_work(self, work_id: str) -> str:
        """Find the creation event ID for a given work."""
        work_name = None
        for entity_id, entity in self.entities_lookup.items():
            if entity_id == work_id:
                work_name = self.urify_name(entity["name"])
                break
        
        if work_name:
            return f"{work_name}_Creation"
        return None
    
    def mint_influence_relations_only(self, group_key: str, event_group: List[Dict[str, Any]], 
                                    entities: Dict[str, Any]):
        """Mint influenced_by relations, consolidating them with creation events."""
        if group_key in self.processed_events:
            return
            
        # Store entities lookup for creation event finding
        self.entities_lookup = entities
        
        # Collect all influences for consolidation
        creation_influences = []
        other_influences = []
        
        # Process each influenced_by relation
        for relation in event_group:
            if relation["relation_type"] == "influenced_by":
                # Correct polarity based on domain/range rules
                corrected_source_id, corrected_target_id = self.correct_influence_polarity(relation, entities)
                
                if not corrected_source_id or not corrected_target_id:
                    continue
                    
                source_entity = entities.get(corrected_source_id)
                target_entity = entities.get(corrected_target_id)
                
                if source_entity and target_entity:
                    source_name = self.urify_name(source_entity["name"])
                    target_name = self.urify_name(target_entity["name"])
                    
                    # If source is a work, connect the Creation event instead
                    if source_entity["type"] == "work":
                        creation_event_id = self.find_creation_event_for_work(corrected_source_id)
                        if creation_event_id:
                            creation_influences.append(f"    {self.crm}P15_was_influenced_by {self.ex}{target_name}")
                    # If source is the creator person, redirect to creation event
                    elif (hasattr(self, 'current_creator_name') and 
                          source_name == self.current_creator_name):
                        creation_influences.append(f"    {self.crm}P15_was_influenced_by {self.ex}{target_name}")
                    else:
                        # Other person influences remain direct
                        other_influences.append(f"{self.ex}{source_name} {self.crm}P15_was_influenced_by {self.ex}{target_name} .")
        
        # Update the creation event with consolidated influences
        if creation_influences and hasattr(self, 'current_creation_id'):
            # Find and update the creation event in rdf_content
            creation_line_idx = None
            for i, line in enumerate(self.rdf_content):
                if f"{self.ex}{self.current_creation_id} a {self.crm}E65_Creation" in line:
                    creation_line_idx = i
                    break
            
            if creation_line_idx is not None:
                # Find the end of the creation event (line ending with '.')
                end_idx = creation_line_idx
                for i in range(creation_line_idx, len(self.rdf_content)):
                    if self.rdf_content[i].strip().endswith('.'):
                        end_idx = i
                        break
                
                # Remove the final '.' and add influences
                self.rdf_content[end_idx] = self.rdf_content[end_idx].rstrip(' .') + ' ;'
                
                # Add influences
                for i, influence in enumerate(creation_influences):
                    if i < len(creation_influences) - 1:
                        self.rdf_content.insert(end_idx + 1 + i, f"{influence} ;")
                    else:
                        self.rdf_content.insert(end_idx + 1 + i, f"{influence} .")
        
        # Add other influences separately
        self.rdf_content.extend(other_influences)
        
        # Mark as processed
        self.processed_events.add(group_key)

    def mint_influence_relations(self, group_key: str, event_group: List[Dict[str, Any]], 
                               entities: Dict[str, Any]):
        """Mint influenced_by relations using P15_influenced_by with correct polarity and event targeting."""
        # Add all involved entities first
        for entity in entities.values():
            self.add_entity_rdf(entity)
        
        # Then create the relations
        self.mint_influence_relations_only(group_key, event_group, entities)
        
    def mint_location_relations_only(self, group_key: str, event_group: List[Dict[str, Any]],
                                   entities: Dict[str, Any]):
        """Mint location relations using P53_has_former_or_current_location and P4_has_time-span."""
        if group_key in self.processed_events:
            return
            
        # Process each location relation
        for relation in event_group:
            rel_type = relation["relation_type"]
            source_id = relation["source_id"]
            target_id = relation["target_id"]
            
            source_entity = entities.get(source_id)
            target_entity = entities.get(target_id)
            
            if not source_entity or not target_entity:
                continue
                
            source_name = self.urify_name(source_entity["name"])
            target_name = self.urify_name(target_entity["name"])
            
            # Handle spatial location
            if rel_type == "located_in_space" and target_entity["type"] == "place":
                if source_entity["type"] == "work":
                    # Redirect work spatial location to the Creation event as P7_took_place_at
                    creation_id = f"{source_name}_Creation"
                    # Check if Creation event exists in rdf_content
                    creation_line_idx = None
                    for i, line in enumerate(self.rdf_content):
                        if f"{self.ex}{creation_id} a {self.crm}E65_Creation" in line:
                            creation_line_idx = i
                            break
                    if creation_line_idx is None:
                        # Mint a minimal Creation event block and attach place
                        self.rdf_content.append(f"{self.ex}{creation_id} a {self.crm}E65_Creation ;")
                        self.rdf_content.append(f"    {self.crm}P94_has_created {self.ex}{source_name} ;")
                        self.rdf_content.append(f"    {self.crm}P7_took_place_at {self.ex}{target_name} .")
                        self.rdf_content.append("")
                    else:
                        # Append P7 to existing Creation event block
                        end_idx = creation_line_idx
                        for j in range(creation_line_idx, len(self.rdf_content)):
                            if self.rdf_content[j].strip().endswith('.'):
                                end_idx = j
                                break
                        self.rdf_content[end_idx] = self.rdf_content[end_idx].rstrip(' .') + ' ;'
                        self.rdf_content.insert(end_idx + 1, f"    {self.crm}P7_took_place_at {self.ex}{target_name} .")
                else:
                    # Use the pattern for non-work entities (e.g., people)
                    location_triple = pattern_spatial_location(self.ex, self.crm, source_name, target_name)
                    self.rdf_content.append(location_triple)
                
            # Handle temporal location
            elif rel_type == "located_in_time" and target_entity["type"] == "date":
                # Use the pattern function for temporal location
                temporal_triple = pattern_temporal_location(self.ex, self.crm, source_name, target_name)
                self.rdf_content.append(temporal_triple)
        
        # Mark as processed
        self.processed_events.add(group_key)
        
    def mint_location_relations(self, group_key: str, event_group: List[Dict[str, Any]],
                              entities: Dict[str, Any]):
        """Mint location relations with spatial and temporal properties."""
        # Add all involved entities first
        for entity in entities.values():
            self.add_entity_rdf(entity)
        
        # Then create the relations
        self.mint_location_relations_only(group_key, event_group, entities)
        
    def mint_association_relations_only(self, group_key: str, event_group: List[Dict[str, Any]],
                                     entities: Dict[str, Any]):
        """Mint association relations using P107_has_current_or_former_member or P67_refers_to."""
        if group_key in self.processed_events:
            return
            
        # Process each association/refers_to relation
        for relation in event_group:
            rel_type = relation["relation_type"]
            source_id = relation["source_id"]
            target_id = relation["target_id"]
            
            source_entity = entities.get(source_id)
            target_entity = entities.get(target_id)
            
            if not source_entity or not target_entity:
                continue
            
            source_name = self.urify_name(source_entity["name"])
            target_name = self.urify_name(target_entity["name"])
            
            # Handle membership-only semantics for associated_with
            if rel_type == "associated_with":
                if target_entity["type"] in ["organization"] and source_entity["type"] in ["person"]:
                    membership_triple = pattern_membership(self.ex, self.crm, source_name, target_name)
                    self.rdf_content.append(membership_triple)
                elif source_entity["type"] == "work":
                    # Backward-compatibility: map legacy work-associated_with-X to refers_to (P67)
                    association_triple = pattern_association(self.ex, self.crm, source_name, target_name)
                    self.rdf_content.append(association_triple)
                else:
                    # Skip non-membership associated_with
                    continue
            # Explicit refers_to: only emit P67 from works to any other entity
            elif rel_type == "refers_to":
                if source_entity["type"] == "work":
                    association_triple = pattern_association(self.ex, self.crm, source_name, target_name)
                    self.rdf_content.append(association_triple)
                else:
                    # Skip non-work sources for refers_to
                    continue
        
        # Mark as processed
        self.processed_events.add(group_key)
        
    def mint_association_relations(self, group_key: str, event_group: List[Dict[str, Any]],
                                entities: Dict[str, Any]):
        """Mint association relations with appropriate properties."""
        # Add all involved entities first
        for entity in entities.values():
            self.add_entity_rdf(entity)
        
        # Then create the relations
        self.mint_association_relations_only(group_key, event_group, entities)
        
    def mint_person_language_relations_only(self, group_key: str, event_group: List[Dict[str, Any]],
                                         entities: Dict[str, Any]):
        """Mint person language relations using P2_has_type with language speaker type."""
        if group_key in self.processed_events:
            return
            
        
        
        # Process each language relation
        for relation in event_group:
            rel_type = relation["relation_type"]
            source_id = relation["source_id"]
            target_id = relation["target_id"]
            
            
            
            source_entity = entities.get(source_id)
            target_entity = entities.get(target_id)
            
            if not source_entity or not target_entity:
                continue
                
            person_name = self.urify_name(source_entity["name"])
            language_name = self.urify_name(target_entity["name"])
            
            
            
            # Handle language competence
            if rel_type == "speaks_language" and target_entity["type"] == "language":
                # Create language speaker type with appellation
                language_speaker_triples = pattern_language_speaker(self.ex, self.crm, person_name, language_name, target_entity["name"])
                
                # Debug: Check if Old French is in the RDF content
                old_french_in_rdf = False
                for line in self.rdf_content:
                    if "Old_French" in line:
                        old_french_in_rdf = True
                if not old_french_in_rdf:
                    # Add Old French entity explicitly
                    self.rdf_content.append(f"ex:{language_name} a crm:E56_Language ;")
                    self.rdf_content.append(f'    rdfs:label "{target_entity["name"]}" .')
                    self.rdf_content.append("")
                
                # Add only the language speaker type (no P72_has_language on persons)
                self.rdf_content.extend(language_speaker_triples)
            
        
        
        # Mark as processed
        self.processed_events.add(group_key)
        
    def mint_person_language_relations(self, group_key: str, event_group: List[Dict[str, Any]],
                                    entities: Dict[str, Any]):
        """Mint person language relations with appropriate properties."""
        
        
        # Add all involved entities first
        for entity in entities.values():
            self.add_entity_rdf(entity)
        
        
        # Then create the relations
        self.mint_person_language_relations_only(group_key, event_group, entities)
        
    def mint_person_expertise_relations_only(self, group_key: str, event_group: List[Dict[str, Any]],
                                          entities: Dict[str, Any]):
        """Mint person expertise relations using P2_has_type with expertise type."""
        if group_key in self.processed_events:
            return
            
        # Process each expertise relation
        for relation in event_group:
            rel_type = relation["relation_type"]
            source_id = relation["source_id"]
            target_id = relation["target_id"]
            
            source_entity = entities.get(source_id)
            target_entity = entities.get(target_id)
            
            if not source_entity or not target_entity:
                continue
                
            person_name = self.urify_name(source_entity["name"])
            expertise_name = self.urify_name(target_entity["name"])
            
            # Handle expertise
            if rel_type == "has_expertise_in" and target_entity["type"] in ["concept", "language", "genre"]:
                # Create expertise type and assign to person (use slug for ID, original name for label)
                expertise_triples = pattern_expertise(self.ex, self.crm, person_name, expertise_name, target_entity["name"])
                for triple in expertise_triples:
                    self.rdf_content.append(triple)
        
        # Mark as processed
        self.processed_events.add(group_key)
        
    def mint_person_expertise_relations(self, group_key: str, event_group: List[Dict[str, Any]],
                                     entities: Dict[str, Any]):
        """Mint person expertise relations with appropriate properties."""
        # Add all involved entities first
        for entity in entities.values():
            self.add_entity_rdf(entity)
        
        # Then create the relations
        self.mint_person_expertise_relations_only(group_key, event_group, entities)

    def mint_person_birth_event_only(self, group_key: str, event_group: List[Dict[str, Any]], 
                                     entities: Dict[str, Any]):
        """Mint a Birth event (E67_Birth) with P98_brought_into_life and optional P4 time-span and P7 place."""
        if group_key in self.processed_events:
            return
        # Identify central person and optional time/place
        person_entity = None
        time_entity = None
        place_entity = None
        for relation in event_group:
            rel_type = relation["relation_type"]
            source_id = relation["source_id"]
            target_id = relation["target_id"]
            # Source is the person for birth relations
            if not person_entity:
                person_entity = entities.get(source_id)
            if rel_type == "date_of_birth":
                time_entity = entities.get(target_id)
            elif rel_type == "place_of_birth":
                place_entity = entities.get(target_id)
        if not person_entity:
            return
        person_name = self.urify_name(person_entity["name"])
        event_id = f"{person_name}_Birth"
        # Build event block
        props = [
            f"    {self.crm}P98_brought_into_life {self.ex}{person_name}"
        ]
        if time_entity:
            props.append(f"    {self.crm}P4_has_time-span {self.ex}{self.urify_name(time_entity['name'])}")
        if place_entity:
            props.append(f"    {self.crm}P7_took_place_at {self.ex}{self.urify_name(place_entity['name'])}")
        # Emit
        self.rdf_content.append(f"{self.ex}{event_id} a {self.crm}E67_Birth ;")
        for i, prop in enumerate(props):
            if i < len(props) - 1:
                self.rdf_content.append(f"{prop} ;")
            else:
                self.rdf_content.append(f"{prop} .")
        self.rdf_content.append("")
        self.processed_events.add(group_key)

    def mint_person_death_event_only(self, group_key: str, event_group: List[Dict[str, Any]], 
                                     entities: Dict[str, Any]):
        """Mint a Death event (E69_Death) with P100_was_death_of and optional P4 time-span and P7 place."""
        if group_key in self.processed_events:
            return
        # Identify central person and optional time/place
        person_entity = None
        time_entity = None
        place_entity = None
        for relation in event_group:
            rel_type = relation["relation_type"]
            source_id = relation["source_id"]
            target_id = relation["target_id"]
            # Source is the person for death relations
            if not person_entity:
                person_entity = entities.get(source_id)
            if rel_type == "date_of_death":
                time_entity = entities.get(target_id)
            elif rel_type == "place_of_death":
                place_entity = entities.get(target_id)
        if not person_entity:
            return
        person_name = self.urify_name(person_entity["name"])
        event_id = f"{person_name}_Death"
        # Build event block
        props = [
            f"    {self.crm}P100_was_death_of {self.ex}{person_name}"
        ]
        if time_entity:
            props.append(f"    {self.crm}P4_has_time-span {self.ex}{self.urify_name(time_entity['name'])}")
        if place_entity:
            props.append(f"    {self.crm}P7_took_place_at {self.ex}{self.urify_name(place_entity['name'])}")
        # Emit
        self.rdf_content.append(f"{self.ex}{event_id} a {self.crm}E69_Death ;")
        for i, prop in enumerate(props):
            if i < len(props) - 1:
                self.rdf_content.append(f"{prop} ;")
            else:
                self.rdf_content.append(f"{prop} .")
        self.rdf_content.append("")
        self.processed_events.add(group_key)

    def generate_grouped_rdf_content(self, entities_lookup: Dict[str, Any], event_groups: Dict[str, List[Dict[str, Any]]], include_prefixes: bool = True, emit_entities: bool = True) -> List[str]:
        """Generate RDF content with entities grouped with their properties.
        Set include_prefixes=False to omit @prefix lines for embedding inside a named graph.
        """
        grouped_content = []
        if include_prefixes:
            grouped_content.extend([
                "@prefix crm: <http://www.cidoc-crm.org/cidoc-crm/> .",
                "@prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .", 
                "@prefix ex: <http://example.org/> .",
                ""
            ])
        
        if emit_entities:
            # First pass: collect all entities that will be used
            all_used_entities = set()
            for event_group in event_groups.values():
                for relation in event_group:
                    all_used_entities.add(relation["source_id"])
                    all_used_entities.add(relation["target_id"])
            
            # Group entities by type for better organization
            entities_by_type = {}
            for entity_id in all_used_entities:
                if entity_id in entities_lookup:
                    entity = entities_lookup[entity_id]
                    entity_type = entity["type"]
                    if entity_type not in entities_by_type:
                        entities_by_type[entity_type] = []
                    entities_by_type[entity_type].append(entity)
            
            # Output entities grouped by type
            type_order = ["work", "person", "organization", "place", "concept", "language", "historical_context", "date"]
            
            for entity_type in type_order:
                if entity_type in entities_by_type:
                    for entity in entities_by_type[entity_type]:
                        entity_id = entity["id"]
                        entity_name = entity["name"]
                        urified_name = self.urify_name(entity_name)
                        cidoc_class = self.entity_mappings.get(entity_type, "E1_CRM_Entity")
                        
                        # Add entity with appellation if needed
                        if entity_type in ["work", "person", "place", "organization"]:
                            appellation_id = f"{urified_name}_appellation"
                            grouped_content.append(f"{self.ex}{urified_name} a {self.crm}{cidoc_class} ;")
                            grouped_content.append(f"    {self.crm}P1_is_identified_by {self.ex}{appellation_id} .")
                            grouped_content.append("")
                            grouped_content.append(f"{self.ex}{appellation_id} a {self.crm}E41_Appellation ;")
                            grouped_content.append(f'    rdfs:label "{entity_name}" .')
                        else:
                            grouped_content.append(f"{self.ex}{urified_name} a {self.crm}{cidoc_class} ;")
                            grouped_content.append(f'    rdfs:label "{entity_name}" .')
                        
                        grouped_content.append("")
                        self.processed_entities.add(entity_id)
        
        return grouped_content
    
    def generate_event_rdf(self, work_schema_file: str) -> str:
        """Generate complete event-centric CIDOC-CRM RDF."""
        data = self.load_work_schema(work_schema_file)
        interpretation_layer = data["work_schema_metadata"]["interpretation_layer"]
        
        nodes = interpretation_layer["nodes"]
        relations = interpretation_layer["relations"]
        
        # Create entity lookup
        entities_lookup = {entity["id"]: entity for entity in nodes}
        
        # Debug check for specific entities
        if "old_french" in entities_lookup:
            print(f"\nDEBUG: Old French entity found in entities_lookup: {entities_lookup['old_french']}")
        else:
            print(f"\nDEBUG: Old French entity NOT found in entities_lookup")
            # Check all entity IDs
            print(f"DEBUG: Available entity IDs: {list(entities_lookup.keys())}")
            # Check if there's any language entity
            language_entities = [entity for entity in nodes if entity["type"] == "language"]
            print(f"DEBUG: Language entities: {language_entities}")
        
        # Group relations by event type
        event_groups = self.group_relations_by_event(relations)
        
        # Generate grouped entity content first
        self.rdf_content = self.generate_grouped_rdf_content(entities_lookup, event_groups, include_prefixes=True)
        
        # Add events and relations section
        self.rdf_content.append("")
        
        # Process each event group
        for group_key, event_group in event_groups.items():
            # Determine event type by matching known prefixes to avoid underscore ambiguity
            event_type = None
            for et in self.event_clusters.keys():
                if group_key.startswith(f"{et}_"):
                    event_type = et
                    break
            if event_type is None:
                event_type = group_key.split("_", 1)[0]
            print(f"Processing event group: {group_key}, type: {event_type}, relations: {len(event_group)}")
            
            # Collect entities for this event
            involved_entities = self.collect_entities_for_event_group(event_group, entities_lookup)
            
            # Mint appropriate event type (entities already added)
            if event_type == "creation":
                self.mint_creation_event_only(group_key, event_group, involved_entities)
            elif event_type == "influence":
                self.mint_influence_relations_only(group_key, event_group, involved_entities)
            elif event_type == "person_language":
                # Process language relations directly here (without debug prints)
                # Add all involved entities first
                for entity_id, entity in involved_entities.items():
                    self.add_entity_rdf(entity)
                # Process each language relation
                for relation in event_group:
                    if relation["relation_type"] == "speaks_language":
                        source_id = relation["source_id"]
                        target_id = relation["target_id"]
                        source_entity = involved_entities.get(source_id)
                        target_entity = involved_entities.get(target_id)
                        if source_entity and target_entity and target_entity["type"] == "language":
                            source_name = self.urify_name(source_entity["name"])
                            language_name = self.urify_name(target_entity["name"])
                            # Add language entity if not already in RDF content
                            language_entity_found = any(f"ex:{language_name} a" in line for line in self.rdf_content)
                            if not language_entity_found:
                                self.rdf_content.append(f"ex:{language_name} a crm:E56_Language ;")
                                self.rdf_content.append(f'    rdfs:label "{target_entity["name"]}" .')
                                self.rdf_content.append("")
                            # Add only the language speaker type (no P72_has_language)
                            speaker_triples = pattern_language_speaker(self.ex, self.crm, source_name, language_name, target_entity["name"])
                            self.rdf_content.extend(speaker_triples)
                
                # Mark as processed
                self.processed_events.add(group_key)
            elif event_type == "person_expertise":
                self.mint_person_expertise_relations(group_key, event_group, involved_entities)
            elif event_type == "person_occupation":
                self.mint_person_occupation_relations(group_key, event_group, involved_entities)
            elif event_type == "person_birth":
                # Entities already emitted at top; mint event only
                self.mint_person_birth_event_only(group_key, event_group, involved_entities)
            elif event_type == "person_death":
                self.mint_person_death_event_only(group_key, event_group, involved_entities)
            elif event_type.startswith("person_"):
                # Other person groups not implemented
                print(f"Skipping person relation group: {group_key} (not yet implemented)")
                self.processed_events.add(group_key)
            elif event_type == "association":
                self.mint_association_relations(group_key, event_group, involved_entities)
            elif event_type == "location":
                self.mint_location_relations(group_key, event_group, involved_entities)
            # TODO: Add other event types
        
        return "\n".join(self.rdf_content)

    def generate_event_rdf_from_data(self, data: Dict[str, Any], include_prefixes: bool = False, emit_entities: bool = True) -> str:
        """Generate event-centric RDF from an in-memory work schema dict.
        When embedding inside a TRIG named graph, set include_prefixes=False.
        """
        interpretation_layer = data["work_schema_metadata"]["interpretation_layer"]
        nodes = interpretation_layer["nodes"]
        relations = interpretation_layer["relations"]

        entities_lookup = {entity["id"]: entity for entity in nodes}
        event_groups = self.group_relations_by_event(relations)
        # reset per-run state
        self.rdf_content = self.generate_grouped_rdf_content(entities_lookup, event_groups, include_prefixes=include_prefixes, emit_entities=emit_entities)
        self.rdf_content.append("")

        for group_key, event_group in event_groups.items():
            event_type = None
            for et in self.event_clusters.keys():
                if group_key.startswith(f"{et}_"):
                    event_type = et
                    break
            if event_type is None:
                event_type = group_key.split("_", 1)[0]
            involved_entities = self.collect_entities_for_event_group(event_group, entities_lookup)
            if event_type == "creation":
                self.mint_creation_event_only(group_key, event_group, involved_entities)
            elif event_type == "influence":
                self.mint_influence_relations_only(group_key, event_group, involved_entities)
            elif event_type == "person_language":
                if emit_entities:
                    for entity_id, entity in involved_entities.items():
                        self.add_entity_rdf(entity)
                for relation in event_group:
                    if relation["relation_type"] == "speaks_language":
                        source_id = relation["source_id"]
                        target_id = relation["target_id"]
                        source_entity = involved_entities.get(source_id)
                        target_entity = involved_entities.get(target_id)
                        if source_entity and target_entity and target_entity["type"] == "language":
                            source_name = self.urify_name(source_entity["name"])
                            language_name = self.urify_name(target_entity["name"])
                            if emit_entities:
                                language_entity_found = any(f"ex:{language_name} a" in line for line in self.rdf_content)
                                if not language_entity_found:
                                    self.rdf_content.append(f"ex:{language_name} a crm:E56_Language ;")
                                    self.rdf_content.append(f'    rdfs:label "{target_entity["name"]}" .')
                                    self.rdf_content.append("")
                            # Add only the language speaker type (no P72_has_language)
                            speaker_triples = pattern_language_speaker(self.ex, self.crm, source_name, language_name, target_entity["name"])
                            self.rdf_content.extend(speaker_triples)
                self.processed_events.add(group_key)
            elif event_type == "person_expertise":
                self.mint_person_expertise_relations_only(group_key, event_group, involved_entities)
            elif event_type == "person_occupation":
                self.mint_person_occupation_relations_only(group_key, event_group, involved_entities)
            elif event_type == "person_birth":
                self.mint_person_birth_event_only(group_key, event_group, involved_entities)
            elif event_type == "person_death":
                self.mint_person_death_event_only(group_key, event_group, involved_entities)
            elif event_type == "association":
                self.mint_association_relations_only(group_key, event_group, involved_entities)
            elif event_type == "location":
                self.mint_location_relations_only(group_key, event_group, involved_entities)

        return "\n".join(self.rdf_content)

    def mint_person_occupation_relations_only(self, group_key: str, event_group: List[Dict[str, Any]],
                                              entities: Dict[str, Any]):
        """Mint occupation as E7_Activity identified by an appellation (occupation label),
        linked with P14_carried_out_by to the person."""
        if group_key in self.processed_events:
            return
        for relation in event_group:
            if relation["relation_type"] != "has_occupation":
                continue
            source = entities.get(relation["source_id"])  # person
            target = entities.get(relation["target_id"])  # occupation role/concept label
            if not source or not target:
                continue
            if source.get("type") != "person":
                continue
            person_name = self.urify_name(source["name"])
            occ_label = target["name"]
            occ_slug = self.urify_name(occ_label)
            activity_id = f"{person_name}_{occ_slug}_Activity"
            app_id = f"{activity_id}_appellation"
            # Emit E7 Activity block with P14 and P1 identified by an appellation labeled as occupation
            self.rdf_content.append(f"{self.ex}{activity_id} a {self.crm}E7_Activity ;")
            self.rdf_content.append(f"    {self.crm}P14_carried_out_by {self.ex}{person_name} ;")
            self.rdf_content.append(f"    {self.crm}P1_is_identified_by {self.ex}{app_id} .")
            self.rdf_content.append("")
            self.rdf_content.append(f"{self.ex}{app_id} a {self.crm}E41_Appellation ;")
            self.rdf_content.append(f"    rdfs:label \"{occ_label}\" .")
            self.rdf_content.append("")
        self.processed_events.add(group_key)

    def mint_person_occupation_relations(self, group_key: str, event_group: List[Dict[str, Any]],
                                         entities: Dict[str, Any]):
        """Add involved entities then emit occupation activities."""
        for entity in entities.values():
            self.add_entity_rdf(entity)
        self.mint_person_occupation_relations_only(group_key, event_group, entities)

    def save_rdf_to_file(self, output_file: str):
        """Save RDF content to file."""
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write("\n".join(self.rdf_content))

def main():
    """Generate TRIG per document with facts and assertions graphs and nanopub head."""
    documents_dir = "./documents"
    os.makedirs(documents_dir, exist_ok=True)
    document_dirs = [d for d in os.listdir(documents_dir) if os.path.isdir(os.path.join(documents_dir, d))]
    if not document_dirs:
        print(f"No document directories found in {documents_dir}")
        return

    total_processed = 0
    errors = []
    for doc_dir in document_dirs:
        doc_path = os.path.join(documents_dir, doc_dir)
        relations_file = os.path.join(doc_path, "relations.json")
        out_trig = os.path.join(doc_path, "cidoc.trig")
        if not os.path.exists(relations_file):
            print(f"No relations.json found in {doc_path}")
            continue
        try:
            with open(relations_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            interp = data.get("work_schema_metadata", {}).get("interpretation_layer", {})
            nodes = interp.get("nodes", [])
            relations = interp.get("relations", [])
            # Split by claim_type
            facts = [r for r in relations if r.get("claim_type") == "established_fact"]
            assertions = [r for r in relations if r.get("claim_type") == "authorial_argument"]
            # Prepare two in-memory schemas
            base_ws = {"work_schema_metadata": {"interpretation_layer": {"nodes": nodes, "relations": []}}}
            facts_ws = json.loads(json.dumps(base_ws))
            facts_ws["work_schema_metadata"]["interpretation_layer"]["relations"] = facts
            assertions_ws = json.loads(json.dumps(base_ws))
            assertions_ws["work_schema_metadata"]["interpretation_layer"]["relations"] = assertions
            # Build an entity block using ALL relations (facts + assertions) so all URIs are defined in facts
            combined_ws = json.loads(json.dumps(base_ws))
            combined_ws["work_schema_metadata"]["interpretation_layer"]["relations"] = relations
            gen_entities = CidocEventGroupGenerator()
            entity_block = gen_entities.generate_grouped_rdf_content(
                {entity["id"]: entity for entity in nodes},
                gen_entities.group_relations_by_event(combined_ws["work_schema_metadata"]["interpretation_layer"]["relations"]),
                include_prefixes=False,
                emit_entities=True,
            )
            entity_block_str = "\n".join(entity_block)
            # Generate facts events without entities
            gen_facts = CidocEventGroupGenerator()
            facts_events = gen_facts.generate_event_rdf_from_data(facts_ws, include_prefixes=False, emit_entities=False)
            facts_content = (entity_block_str + ("\n" if entity_block_str.strip() and facts_events.strip() else "") + facts_events).strip()
            # Generate assertions without entities
            gen_assert = CidocEventGroupGenerator()
            assertion_content = gen_assert.generate_event_rdf_from_data(assertions_ws, include_prefixes=False, emit_entities=False)

            # Compose TRIG
            trig_lines = [
                "@prefix crm: <http://www.cidoc-crm.org/cidoc-crm/> .",
                "@prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .",
                "@prefix ex: <http://example.org/> .",
                "@prefix np: <http://www.nanopub.org/nschema#> .",
                ""
            ]
            # Named graphs
            trig_lines.append(f"ex:facts_{doc_dir} {{")
            if facts_content.strip():
                trig_lines.append(facts_content)
            trig_lines.append("}")
            trig_lines.append("")
            trig_lines.append(f"ex:assertion_{doc_dir} {{")
            if assertion_content.strip():
                trig_lines.append(assertion_content)
            trig_lines.append("}")
            trig_lines.append("")
            # Head graph
            trig_lines.append(f"ex:head_{doc_dir} {{")
            trig_lines.append(f"    ex:pub_{doc_dir} a np:Nanopublication ;")
            trig_lines.append(f"        np:hasAssertion ex:assertion_{doc_dir} ;")
            trig_lines.append(f"        np:hasProvenance ex:provenance_{doc_dir} ;")
            trig_lines.append(f"        np:hasPublicationInfo ex:pubInfo_{doc_dir} .")
            trig_lines.append("}")

            with open(out_trig, 'w', encoding='utf-8') as outf:
                outf.write("\n".join(trig_lines))
            print(f"Wrote {out_trig}")
            total_processed += 1
        except Exception as e:
            msg = f"Error processing {relations_file}: {e}"
            print(msg)
            errors.append(msg)

    print(f"\nProcessing complete. {total_processed} files processed.")
    if errors:
        print("Errors:\n- " + "\n- ".join(errors))

if __name__ == "__main__":
    main()
