"""
Script to completely clean the Neo4j database.
WARNING: This will DELETE ALL DATA in the database!
"""

from neo4j import GraphDatabase

# Neo4j Connection
uri = "bolt://localhost:7687"
username = "neo4j"
password = "Anjitha@2002"

driver = GraphDatabase.driver(uri, auth=(username, password))

print("=" * 70)
print("NEO4J DATABASE CLEANUP")
print("=" * 70)

try:
    with driver.session() as session:
        print("\n🗑️  Deleting all nodes and relationships...")
        session.run("MATCH (n) DETACH DELETE n")
        print("✅ All nodes and relationships deleted!")
        
        print("\n📊 Database Statistics After Cleanup:")
        result = session.run("MATCH (n) RETURN COUNT(n) AS node_count")
        node_count = result.single()["node_count"]
        print(f"   Total Nodes: {node_count}")
        
        result = session.run("MATCH ()-[r]-() RETURN COUNT(r) AS rel_count")
        rel_count = result.single()["rel_count"]
        print(f"   Total Relationships: {rel_count}")
        
    print("\n✅ Database cleaned successfully!")
    print("✅ Ready to ingest fresh data with: python free_kg.py")
    
except Exception as e:
    print(f"\n❌ Error during cleanup: {e}")

finally:
    driver.close()

print("\n" + "=" * 70)
