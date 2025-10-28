#!/usr/bin/env python3
"""
Script to count RDF triples in nanopublication layers.
Counts triples in: factual graph (Layer 0), assertion graph (Layer 1), 
and provenance graph (Layer 2), excluding publication info.
"""

from rdflib import Dataset, Namespace
from collections import defaultdict
import sys

def load_nanopub(file_path):
    """Load a nanopublication from a file."""
    d = Dataset()
    d.parse(file_path, format='trig')
    return d

def identify_graphs(dataset):
    """
    Identify the different named graphs in the nanopublication.
    Returns dictionaries with graph URIs for facts, assertion, provenance, and pubInfo.
    """
    graphs = {'facts': None, 'assertion': None, 'provenance': None, 'pubInfo': None, 'head': None}
    
    for context in dataset.graphs():
        graph_uri = str(context.identifier)
        
        # Skip the default graph
        if graph_uri == 'urn:x-rdflib:default':
            continue
        
        # Identify graph type based on naming conventions
        if 'facts' in graph_uri.lower():
            graphs['facts'] = context.identifier
        elif 'assertion' in graph_uri.lower():
            graphs['assertion'] = context.identifier
        elif 'provenance' in graph_uri.lower():
            graphs['provenance'] = context.identifier
        elif 'pubinfo' in graph_uri.lower():
            graphs['pubInfo'] = context.identifier
        elif 'head' in graph_uri.lower():
            graphs['head'] = context.identifier
    
    return graphs

def count_triples_in_graph(dataset, graph_uri):
    """Count triples in a specific named graph."""
    if graph_uri is None:
        return 0
    
    graph = dataset.get_context(graph_uri)
    return len(graph)

def get_nanopub_name(file_path):
    """Extract a readable name from the file path or content."""
    d = Dataset()
    d.parse(file_path, format='trig')
    
    # Try to find the source article title or identifier
    query = """
    PREFIX dcterms: <http://purl.org/dc/terms/>
    PREFIX fabio: <http://purl.org/spar/fabio/>
    
    SELECT ?title
    WHERE {
        GRAPH ?g {
            ?article a ?type .
            ?article dcterms:title ?title .
            FILTER(?type IN (fabio:JournalArticle, fabio:Book, fabio:BookChapter))
        }
    }
    LIMIT 1
    """
    
    for row in d.query(query):
        # Return abbreviated version of title
        title = str(row.title)
        if 'robotfoto' in title.lower():
            return 'Van Daele (2005)'
        elif 'reynaert the fox' in title.lower():
            return 'Besamusca & Bouwman (2009)'
        elif 'historiciteit' in title.lower():
            return 'Peeters (1999)'
    
    # Fallback to filename
    return file_path.split('/')[-1].replace('.trig', '').replace('.ttl', '')

def analyze_nanopub(file_path):
    """Analyze a single nanopublication and return statistics."""
    dataset = load_nanopub(file_path)
    
    # Identify the different graphs
    graphs = identify_graphs(dataset)
    
    # Count triples in each graph
    stats = {
        'name': get_nanopub_name(file_path),
        'facts': count_triples_in_graph(dataset, graphs['facts']),
        'assertion': count_triples_in_graph(dataset, graphs['assertion']),
        'provenance': count_triples_in_graph(dataset, graphs['provenance']),
        'pubInfo': count_triples_in_graph(dataset, graphs['pubInfo'])
    }
    
    return stats

def print_statistics(all_stats):
    """Print formatted statistics for all nanopublications."""
    print("\n" + "="*80)
    print("NANOPUBLICATION TRIPLE COUNTS")
    print("="*80)
    print()
    
    # Print header
    print(f"{'Nanopublication':<40} {'Layer 0':<12} {'Layer 1':<12} {'Layer 2':<12} {'Total':<12}")
    print(f"{'(Source Article)':<40} {'(Facts)':<12} {'(Assertion)':<12} {'(Provenance)':<12} {'(No PubInfo)':<12}")
    print("-"*80)
    
    # Print each nanopub
    total_facts = 0
    total_assertions = 0
    total_provenance = 0
    
    for stats in all_stats:
        name = stats['name']
        facts = stats['facts']
        assertion = stats['assertion']
        provenance = stats['provenance']
        total = facts + assertion + provenance
        
        print(f"{name:<40} {facts:<12} {assertion:<12} {provenance:<12} {total:<12}")
        
        total_facts += facts
        total_assertions += assertion
        total_provenance += provenance
    
    # Print totals
    print("-"*80)
    grand_total = total_facts + total_assertions + total_provenance
    print(f"{'TOTAL':<40} {total_facts:<12} {total_assertions:<12} {total_provenance:<12} {grand_total:<12}")
    print()
    
    # Print layer summary
    print("\n" + "="*80)
    print("LAYER SUMMARY")
    print("="*80)
    print(f"Layer 0 (Factual Data Graph):        {total_facts} triples")
    print(f"Layer 1 (Assertion Graph):            {total_assertions} triples")
    print(f"Layer 2 (Hermeneutical Context):      {total_provenance} triples")
    print(f"Total (excluding publication info):   {grand_total} triples")
    print()

def main():
    if len(sys.argv) < 2:
        print("Usage: python count_nanopub_relations.py <file1.trig> [file2.trig] ...")
        print("\nExample: python count_nanopub_relations.py daele_2005.trig besamusca_2009.trig peeters_1999.trig")
        sys.exit(1)
    
    all_stats = []
    
    for file_path in sys.argv[1:]:
        try:
            stats = analyze_nanopub(file_path)
            all_stats.append(stats)
        except Exception as e:
            print(f"Error processing {file_path}: {e}", file=sys.stderr)
    
    if all_stats:
        print_statistics(all_stats)
    else:
        print("No nanopublications were successfully processed.", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()