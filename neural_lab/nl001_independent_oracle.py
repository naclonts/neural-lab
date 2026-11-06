"""Independent bounded NL-001 corpus semantic audit; standard library only.

Does not import generator, checker, parser, or replay. Does not authenticate
corpus provenance, text/query alignment, splits, or presentation identity.
"""
import argparse
import json
from collections import Counter


def _graph(symbols, edges):
    if not isinstance(symbols, list) or any(not isinstance(s, str) or not s for s in symbols) or len(set(symbols)) != len(symbols):
        raise ValueError('invalid symbols')
    if not isinstance(edges, list):
        raise ValueError('invalid edges')
    result = set()
    for e in edges:
        if not isinstance(e, list) or len(e) != 2 or any(not isinstance(x, str) or x not in symbols for x in e):
            raise ValueError('invalid edge')
        result.add(tuple(e))
    if len(result) != len(edges):
        raise ValueError('duplicate edges')
    return result


def truth(symbols, edges, complete, query):
    """Compute nonempty reachability by Warshall closure, including cycles."""
    edge_set = _graph(symbols, edges)
    if type(complete) is not bool:
        raise ValueError('invalid completeness')
    if (not isinstance(query, dict) or set(query) != {'property', 'subject', 'object'} or
        any(not isinstance(v, str) for v in query.values()) or
        query['property'] not in ('calls', 'reachable') or
        query['subject'] not in symbols or query['object'] not in symbols):
        return 'Unknown'
    index = {v: i for i, v in enumerate(symbols)}
    n = len(symbols)
    matrix = [[False] * n for _ in range(n)]
    for a, b in edge_set:
        matrix[index[a]][index[b]] = True
    a, b = index[query['subject']], index[query['object']]
    if query['property'] == 'calls':
        exists = matrix[a][b]
    else:
        for k in range(n):
            for i in range(n):
                for j in range(n):
                    matrix[i][j] = matrix[i][j] or (matrix[i][k] and matrix[k][j])
        exists = matrix[a][b]
    return 'Proven' if exists else ('Refuted' if complete else 'Unknown')


def certificate_valid(symbols, edges, complete, query, label, certificate):
    """Validate evidence independently against graph and query, including closure."""
    expected = truth(symbols, edges, complete, query)
    if label != expected:
        return False
    if label == 'Unknown':
        return certificate is None
    if not isinstance(certificate, dict):
        return False
    edge_set = _graph(symbols, edges)
    a, b, prop = query['subject'], query['object'], query['property']
    if label == 'Proven':
        path = certificate.get('nodes')
        return (set(certificate) == {'kind', 'nodes'} and certificate['kind'] == 'path' and
                isinstance(path, list) and len(path) >= 2 and path[0] == a and path[-1] == b and
                (prop != 'calls' or len(path) == 2) and
                all(isinstance(x, str) and x in symbols for x in path) and
                all((x, y) in edge_set for x, y in zip(path, path[1:])))
    if prop == 'calls':
        return certificate == {'kind': 'absent_edge', 'edge': [a, b]} and (a, b) not in edge_set
    nodes = certificate.get('nodes')
    if (set(certificate) != {'kind', 'nodes'} or certificate['kind'] != 'closed_set' or
        not isinstance(nodes, list) or any(not isinstance(x, str) or x not in symbols for x in nodes) or
        len(nodes) != len(set(nodes)) or a not in nodes):
        return False
    closure = set(nodes)
    if a != b and b in closure:
        return False
    return all(y in closure and y != b for x, y in edge_set if x in closure)


def audit(corpus):
    counts = Counter()
    if not isinstance(corpus, dict) or not isinstance(corpus.get('splits'), dict):
        raise ValueError('invalid corpus')
    for split, entries in corpus['splits'].items():
        for entry in entries:
            for variant in entry['variants']:
                symbols = variant['symbols']
                full = _graph(symbols, variant['full_edges'])
                observed = _graph(symbols, variant['observed_incomplete_edges'])
                removed = _graph(symbols, variant['removed_edges'])
                if not removed or observed & removed or observed | removed != full:
                    raise ValueError('invalid incomplete view')
                for q in variant['questions']:
                    if q['view'] not in ('complete', 'incomplete'):
                        raise ValueError('invalid view')
                    complete = q['view'] == 'complete'
                    edges = variant['full_edges'] if complete else variant['observed_incomplete_edges']
                    if not certificate_valid(symbols, edges, complete, q['query'], q['label'], q['certificate']):
                        raise ValueError('incorrect label or certificate in ' + str(split))
                    counts[(split, q['view'], q['label'])] += 1
    if not counts:
        raise ValueError('empty corpus')
    return {'claims': sum(counts.values()), 'counts': [{'split': k[0], 'view': k[1], 'label': k[2], 'count': v} for k, v in sorted(counts.items())]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('corpus')
    args = parser.parse_args()
    with open(args.corpus, encoding='utf8') as f:
        report = audit(json.load(f))
    print(json.dumps(report, sort_keys=True, indent=2))


if __name__ == '__main__':
    main()
