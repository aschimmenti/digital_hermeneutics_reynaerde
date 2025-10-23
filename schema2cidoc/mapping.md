### Mapping between schema relations and CIDOC-CRM 

| Schema relation | CIDOC-CRM pattern |
| speaks_language | language becomes a "LanguageSpeaker" type (E55_Type), and the person is connected to it through P2_has_type |
| has_expertise_in | expertise becomes a "ExpertiseExpert" type (E55_Type), and the person is connected to it through P2_has_type |
| holds_role | role becomes an activity "RoleActivity" (E7_Activity), and the person is connected to it through P14_carried_out_by in the role of RoleType (E55 Type) |
| educated_at | education becomes an activity "EducationActivity" (E7_Activity), and the person is connected to it through P14_carried_out_by in the role of Learner (E55 Type) |
| member_of | Group =>  P107_has_current_or_former_member => Actor (E39_Actor) / P107i_is_current_or_former_member_of => Group (E74_Group) |
| birth_date | birth_date becomes the timespan of a Birth event (E69_Death), using the property P98_brought_into_life to connect the birth event to the person. timespan and place properties are related |
| birth_place | birth_place becomes the place of a Death event (E69_Death), using the property P100_was_death_of to connect the death event to the person. timespan and place properties are related|
| death_date | death_date becomes the timespan of a Death event (E69_Death), using the property P100_was_death_of to connect the death event to the person. timespan and place properties are related |
| death_place | death_place becomes the place of a Death event (E69_Death), using the property P100_was_death_of to connect the death event to the person. timespan and place properties are related |
| influenced_by | influence connects an activity (like creation event) to any concept through P15_was_influenced_by. It must connect any event related to a person to another event or concept.
| connected_with | this property connects an entity with another in the schema; in CIDOC, it realizes as P107_has_current_or_former_member between a member and an individual which could have as type a Type (E55_Type) or a Group (E74_Group).
| has_occupation | occupation is an Activity event as well. It is a type of activity were the person is connected through P14_carried_out_by in the role of OccupationType (E55 Type).
| lived_in | crm:E21_Person crm:P53_has_former_or_current_location crm:E53_Place. Very simple in this case