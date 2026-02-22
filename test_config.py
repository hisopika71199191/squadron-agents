import os
from squadron.core.config import MemoryConfig

os.environ["NEO4J_PASSWORD"] = "Rr40696003"
config = MemoryConfig()
print(f"Password: {config.neo4j_password}")
