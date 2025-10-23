# CIDOC-CRM Patterns
# Reusable pattern functions for generating CIDOC-CRM triples

def pattern_spatial_location(ex_prefix, crm_prefix, source_name, target_name):
    """Pattern for spatial location using P53_has_former_or_current_location."""
    return f"{ex_prefix}{source_name} {crm_prefix}P53_has_former_or_current_location {ex_prefix}{target_name} ."

def pattern_temporal_location(ex_prefix, crm_prefix, source_name, target_name):
    """Pattern for temporal location using P4_has_time-span."""
    return f"{ex_prefix}{source_name} {crm_prefix}P4_has_time-span {ex_prefix}{target_name} ."

def pattern_type_assignment(ex_prefix, crm_prefix, source_name, type_name):
    """Pattern for type assignment using P2_has_type."""
    return f"{ex_prefix}{source_name} {crm_prefix}P2_has_type {ex_prefix}{type_name} ."

def pattern_influence(ex_prefix, crm_prefix, source_name, target_name):
    """Pattern for influence using P15_was_influenced_by."""
    return f"{ex_prefix}{source_name} {crm_prefix}P15_was_influenced_by {ex_prefix}{target_name} ."

def pattern_membership(ex_prefix, crm_prefix, member_name, group_name):
    """Pattern for group membership using P107_has_current_or_former_member."""
    return f"{ex_prefix}{group_name} {crm_prefix}P107_has_current_or_former_member {ex_prefix}{member_name} ."

def pattern_association(ex_prefix, crm_prefix, source_name, target_name):
    """Pattern for generic association using P67_refers_to."""
    return f"{ex_prefix}{source_name} {crm_prefix}P67_refers_to {ex_prefix}{target_name} ."

def pattern_birth_event(ex_prefix, crm_prefix, person_name, event_id):
    """Pattern for birth event using E67_Birth and P98_brought_into_life."""
    return [
        f"{ex_prefix}{event_id} a {crm_prefix}E67_Birth ;",
        f"    {crm_prefix}P98_brought_into_life {ex_prefix}{person_name} ."
    ]

def pattern_death_event(ex_prefix, crm_prefix, person_name, event_id):
    """Pattern for death event using E69_Death and P100_was_death_of."""
    return [
        f"{ex_prefix}{event_id} a {crm_prefix}E69_Death ;",
        f"    {crm_prefix}P100_was_death_of {ex_prefix}{person_name} ."
    ]

def pattern_activity_participation(ex_prefix, crm_prefix, person_name, activity_id, role_name=None):
    """Pattern for activity participation using E7_Activity and P14_carried_out_by."""
    if role_name:
        return [
            f"{ex_prefix}{activity_id} a {crm_prefix}E7_Activity ;",
            f"    {crm_prefix}P14_carried_out_by {ex_prefix}{person_name} ;",
            f"    {crm_prefix}P2_has_type {ex_prefix}{role_name} ."
        ]
    else:
        return [
            f"{ex_prefix}{activity_id} a {crm_prefix}E7_Activity ;",
            f"    {crm_prefix}P14_carried_out_by {ex_prefix}{person_name} ."
        ]

def pattern_event_place(ex_prefix, crm_prefix, event_id, place_name):
    """Pattern for event place using P7_took_place_at."""
    return f"{ex_prefix}{event_id} {crm_prefix}P7_took_place_at {ex_prefix}{place_name} ;"

def pattern_event_time(ex_prefix, crm_prefix, event_id, time_name):
    """Pattern for event time using P4_has_time-span."""
    return f"{ex_prefix}{event_id} {crm_prefix}P4_has_time-span {ex_prefix}{time_name} ;"

# Language and expertise patterns
def pattern_language_competence(ex_prefix, crm_prefix, person_name, language_name):
    """Pattern for language competence using language type."""
    # Create a simple triple directly connecting person to language
    return f"{ex_prefix}{person_name} {crm_prefix}P72_has_language {ex_prefix}{language_name} ."

def pattern_expertise(ex_prefix, crm_prefix, person_name, expertise_name):
    """Pattern for expertise using E55_Type identified by an appellation labeled '<Expertise> Expert'."""
    expertise_type = f"{expertise_name}_Expert"
    appellation_id = f"{expertise_type}_appellation"
    return [
        f"{ex_prefix}{expertise_type} a {crm_prefix}E55_Type ;",
        f"    {crm_prefix}P1_is_identified_by {ex_prefix}{appellation_id} .",
        f"{ex_prefix}{appellation_id} a {crm_prefix}E41_Appellation ;",
        f'    rdfs:label "{expertise_name} Expert" .',
        f"{ex_prefix}{person_name} {crm_prefix}P2_has_type {ex_prefix}{expertise_type} ."
    ]

def pattern_language_speaker(ex_prefix, crm_prefix, person_name, language_name, language_label=None):
    """Pattern for language speaker type using E55_Type identified by an appellation labeled '<Language> Speaker'."""
    type_id = f"{language_name}_Speaker"
    appellation_id = f"{type_id}_appellation"
    label = f"{language_label or language_name.replace('_',' ')} Speaker"
    return [
        f"{ex_prefix}{type_id} a {crm_prefix}E55_Type ;",
        f"    {crm_prefix}P1_is_identified_by {ex_prefix}{appellation_id} .",
        f"{ex_prefix}{appellation_id} a {crm_prefix}E41_Appellation ;",
        f'    rdfs:label "{label}" .',
        f"{ex_prefix}{person_name} {crm_prefix}P2_has_type {ex_prefix}{type_id} ."
    ]
