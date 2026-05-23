"""
Clean up old COVID-19 test data from Neo4j to avoid confusion with new extractions.
"""
import os
from neo4j import GraphDatabase

# Neo4j connection
uri = "bolt://localhost:7687"
username = "neo4j"
password = "Anjitha@2002"

driver = GraphDatabase.driver(uri, auth=(username, password))

def delete_old_covid_data(tx):
    """Delete COVID-19 related nodes to clean the database."""
    covid_entities = ["COVID-19", "Pfizer", "Moderna", "World Health Organization", "mRNA vaccine", "COVID-19 vaccine", "mRNA vaccine", "vaccine", "2020", "world", "rates", "infection rates", "campaigns", "vaccination campaigns", "Organization"]
    
    # First, let's just get all entities and their relationship counts
    count_query = "MATCH (n:Entity) RETURN n.name, count(*) as count"
    result = list(tx.run(count_query))
    print("Current entities in database:")
    for record in result:
        print(f"  - {record['n.name']}: {record['count']} connections")
    
    # Delete old COVID entities
    delete_query = """
    MATCH (n:Entity)
    WHERE n.name IN $covid_entities
    DETACH DELETE n
    """
    tx.run(delete_query, covid_entities=covid_entities)
    print(f"\n✅ Deleted {len(covid_entities)} old COVID-19 related entities")

with driver.session() as session:
    session.execute_write(delete_old_covid_data)

driver.close()
print("\n✅ Database cleaned. Run free_kg.py again to see fresh Mathematics graph in Explore.")
