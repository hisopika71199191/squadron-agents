import asyncio
import os
from squadron.core.config import MemoryConfig
from squadron.memory.graphiti import GraphitiMemory

async def test_connection():
    config = MemoryConfig(
        neo4j_uri="bolt://localhost:7687",
        neo4j_user="neo4j"
    )
    memory = GraphitiMemory(config=config)
    await memory.initialize()
    
    if memory._graphiti_client:
        print("✅ Connected to Neo4j!")
    else:
        print("⚠️ Using in-memory fallback (Neo4j not connected)")

asyncio.run(test_connection())
