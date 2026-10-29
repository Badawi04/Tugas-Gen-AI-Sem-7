# pip install weaviate-client
"""
Weaviate is an open-source vector database with built-in hybrid search (vector + BM25 keyword),
a GraphQL API, and a Python client.

Start locally:
docker run -d -p 8080:8080 -p 50051:50051 cr.weaviate.io/semitechnologies/weaviate:latest
"""
def print_weaviate_pipeline():
    print("Weaviate patterns:")
    print("1. Schema collection creation: client.collections.create(...)")
    print("2. Hybrid Search: collection.query.hybrid(query='...', vector=[...], alpha=0.5)")
    print("3. Pre-filters: filters=Filter.by_property('category').equal('ai')")

if __name__ == "__main__":
    print_weaviate_pipeline()
