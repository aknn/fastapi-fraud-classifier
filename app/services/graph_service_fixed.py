"""
Graph Service for Entity Relationship Analysis
Manages graph-based fraud detection using NetworkX and Redis
"""
from typing import Dict, List, Any, Set, Tuple, Optional
from datetime import datetime
import aioredis
import networkx as nx
import json

from app.models import GraphQuery
from src.config import settings
from app.database.service import DatabaseService


class GraphService:
    """Manages entity relationship graphs for fraud detection"""

    def __init__(self, db_service: DatabaseService):
        self.db_service = db_service
        self.redis_client: Optional[aioredis.Redis] = None
        self.graph_prefix = "graph:"
        self.node_prefix = "node:"
        self.edge_prefix = "edge:"

        # In-memory graph for complex analysis
        self.transaction_graph = nx.DiGraph()

    async def _get_redis_client(self) -> aioredis.Redis:
        """Get or create async Redis client"""
        if self.redis_client is None:
            self.redis_client = await aioredis.from_url(settings.redis_url)
        return self.redis_client

    async def add_transaction_relationship(
        self,
        user_id: str,
        merchant_id: str,
        device_fingerprint: Optional[str],
        ip_address_hash: Optional[str],
        transaction_amount: float,
        timestamp: datetime
    ) -> Dict[str, Any]:
        """Add transaction relationships to the graph"""

        relationships_created = []

        # User -> Merchant relationship
        if merchant_id:
            relationship = self.db_service.create_entity_relationship(
                source_type="user",
                source_id=user_id,
                target_type="merchant",
                target_id=merchant_id,
                relationship_type="transacts_with",
                weight=transaction_amount,
                extra_data={
                    "last_amount": transaction_amount,
                    "last_timestamp": timestamp.isoformat()
                }
            )
            relationships_created.append("user_merchant")

            # Cache in Redis and update in-memory graph
            await self._cache_relationship(user_id, merchant_id, "user_merchant", transaction_amount)
            self.transaction_graph.add_edge(
                f"user:{user_id}",
                f"merchant:{merchant_id}",
                weight=transaction_amount,
                relationship_type="transacts_with"
            )

        # User -> Device relationship
        if device_fingerprint:
            relationship = self.db_service.create_entity_relationship(
                source_type="user",
                source_id=user_id,
                target_type="device",
                target_id=device_fingerprint,
                relationship_type="uses_device",
                weight=1.0,
                extra_data={
                    "last_seen": timestamp.isoformat()
                }
            )
            relationships_created.append("user_device")

            await self._cache_relationship(user_id, device_fingerprint, "user_device", 1.0)
            self.transaction_graph.add_edge(
                f"user:{user_id}",
                f"device:{device_fingerprint}",
                weight=1.0,
                relationship_type="uses_device"
            )

        # User -> IP relationship (for pattern analysis)
        if ip_address_hash:
            relationship = self.db_service.create_entity_relationship(
                source_type="user",
                source_id=user_id,
                target_type="ip_address",
                target_id=ip_address_hash,
                relationship_type="connects_from",
                weight=1.0,
                extra_data={
                    "last_seen": timestamp.isoformat()
                }
            )
            relationships_created.append("user_ip")

            await self._cache_relationship(user_id, ip_address_hash, "user_ip", 1.0)
            self.transaction_graph.add_edge(
                f"user:{user_id}",
                f"ip:{ip_address_hash}",
                weight=1.0,
                relationship_type="connects_from"
            )

        return {
            "relationships_created": relationships_created,
            "timestamp": timestamp.isoformat(),
            "graph_nodes": self.transaction_graph.number_of_nodes(),
            "graph_edges": self.transaction_graph.number_of_edges()
        }

    async def _cache_relationship(
        self,
        source_id: str,
        target_id: str,
        relationship_type: str,
        weight: float
    ) -> None:
        """Cache relationship in Redis for fast access"""

        redis_client = await self._get_redis_client()

        # Store bidirectional relationships for easy lookup
        forward_key = f"{self.edge_prefix}{source_id}:{relationship_type}"
        backward_key = f"{self.edge_prefix}{target_id}:{relationship_type}_inv"

        edge_data = {
            "target": target_id,
            "weight": weight,
            "timestamp": datetime.utcnow().isoformat(),
            "relationship_type": relationship_type
        }

        await redis_client.lpush(forward_key, json.dumps(edge_data))
        await redis_client.expire(forward_key, 86400 * 30)  # 30 days

        # Reverse relationship
        reverse_edge_data = {
            "target": source_id,
            "weight": weight,
            "timestamp": datetime.utcnow().isoformat(),
            "relationship_type": f"{relationship_type}_inv"
        }

        await redis_client.lpush(backward_key, json.dumps(reverse_edge_data))
        await redis_client.expire(backward_key, 86400 * 30)

    async def find_connected_entities(
        self,
        entity_type: str,
        entity_id: str,
        max_depth: int = 2,
        max_results: int = 100
    ) -> Dict[str, Set[str]]:
        """Find entities connected to the given entity"""

        redis_client = await self._get_redis_client()

        # Try Redis cache first
        cache_key = f"{self.graph_prefix}connected:{entity_type}:{entity_id}:{max_depth}"
        cached_result = await redis_client.get(cache_key)

        if cached_result:
            try:
                cached_data = json.loads(cached_result)
                return {k: set(v) for k, v in cached_data.items()}
            except (json.JSONDecodeError, KeyError):
                pass

        # Compute connections using database
        db_relationships = self.db_service.get_entity_relationships(
            entity_type, entity_id, depth=max_depth)

        connections = {}
        for rel in db_relationships[:max_results]:
            # Determine if this entity is source or target
            if rel.source_type == entity_type and rel.source_id == entity_id:
                # This entity is the source
                target_key = f"{rel.target_type}s"
                if target_key not in connections:
                    connections[target_key] = set()
                connections[target_key].add(rel.target_id)
            elif rel.target_type == entity_type and rel.target_id == entity_id:
                # This entity is the target
                source_key = f"{rel.source_type}s"
                if source_key not in connections:
                    connections[source_key] = set()
                connections[source_key].add(rel.source_id)

        # Cache result
        cacheable_connections = {k: list(v) for k, v in connections.items()}
        await redis_client.setex(
            cache_key, 3600, json.dumps(cacheable_connections))  # 1 hour cache

        return connections

    async def detect_suspicious_patterns(
        self,
        entity_type: str,
        entity_id: str
    ) -> List[Dict[str, Any]]:
        """Detect suspicious patterns in entity relationships"""

        suspicious_patterns = []

        # Get connected entities
        connected = await self.find_connected_entities(entity_type, entity_id, max_depth=2)

        # Pattern 1: Multiple users sharing same device
        if entity_type == "user" and "devices" in connected:
            for device_id in connected["devices"]:
                device_users = await self.find_connected_entities("device", device_id, max_depth=1)
                user_count = len(device_users.get("users", set()))

                if user_count > 5:  # Threshold for suspicious device sharing
                    suspicious_patterns.append({
                        "pattern_type": "device_sharing",
                        "description": f"Device {device_id} used by {user_count} different users",
                        "risk_level": "HIGH" if user_count > 10 else "MEDIUM",
                        "entities_involved": list(device_users.get("users", set())),
                        "confidence": 0.8
                    })

        # Pattern 2: Rapid merchant switching
        if entity_type == "user" and "merchants" in connected:
            merchant_count = len(connected["merchants"])
            if merchant_count > 20:  # Threshold for rapid merchant switching
                suspicious_patterns.append({
                    "pattern_type": "merchant_hopping",
                    "description": f"User transacts with {merchant_count} different merchants",
                    "risk_level": "MEDIUM",
                    "entities_involved": list(connected["merchants"]),
                    "confidence": 0.6
                })

        # Pattern 3: IP address sharing
        if entity_type == "user" and "ip_addresses" in connected:
            for ip_id in connected["ip_addresses"]:
                ip_users = await self.find_connected_entities("ip_address", ip_id, max_depth=1)
                user_count = len(ip_users.get("users", set()))

                if user_count > 3:  # Threshold for suspicious IP sharing
                    suspicious_patterns.append({
                        "pattern_type": "ip_sharing",
                        "description": f"IP address {ip_id} used by {user_count} different users",
                        "risk_level": "MEDIUM",
                        "entities_involved": list(ip_users.get("users", set())),
                        "confidence": 0.7
                    })

        return suspicious_patterns

    async def query_graph(self, query: GraphQuery) -> Dict[str, Any]:
        """Execute graph query and return results"""

        if query.query_type == "shortest_path":
            return await self._find_shortest_path(
                query.source_entity, query.target_entity, query.max_depth or 5)

        elif query.query_type == "connected_components":
            return await self._find_connected_components(
                query.source_entity, query.max_depth or 3)

        elif query.query_type == "community_detection":
            return await self._detect_communities(query.max_depth or 3)

        elif query.query_type == "centrality_analysis":
            return await self._analyze_centrality(query.source_entity)

        else:
            return {"error": f"Unknown query type: {query.query_type}"}

    async def _find_shortest_path(
        self,
        source_entity: str,
        target_entity: str,
        max_depth: int
    ) -> Dict[str, Any]:
        """Find shortest path between two entities"""

        try:
            if source_entity in self.transaction_graph and target_entity in self.transaction_graph:
                path = nx.shortest_path(
                    self.transaction_graph, source_entity, target_entity)
                path_length = len(path) - 1

                return {
                    "path": path,
                    "path_length": path_length,
                    "exists": True
                }
            else:
                return {
                    "path": [],
                    "path_length": -1,
                    "exists": False,
                    "reason": "One or both entities not found in graph"
                }
        except nx.NetworkXNoPath:
            return {
                "path": [],
                "path_length": -1,
                "exists": False,
                "reason": "No path exists"
            }

    async def _find_connected_components(
        self,
        source_entity: str,
        max_depth: int
    ) -> Dict[str, Any]:
        """Find connected component containing the source entity"""

        if source_entity not in self.transaction_graph:
            return {"component": [], "size": 0}

        # Convert to undirected for component analysis
        undirected_graph = self.transaction_graph.to_undirected()

        # Find connected component containing the source
        component = nx.node_connected_component(
            undirected_graph, source_entity)

        return {
            "component": list(component),
            "size": len(component)
        }

    async def _detect_communities(self, max_depth: int) -> Dict[str, Any]:
        """Detect communities in the transaction graph"""

        if self.transaction_graph.number_of_nodes() < 5:
            return {"communities": [], "modularity": 0.0}

        try:
            # Convert to undirected for community detection
            undirected_graph = self.transaction_graph.to_undirected()

            # Simple community detection using connected components
            # In production, would use more sophisticated algorithms
            communities = list(nx.connected_components(undirected_graph))

            return {
                "communities": [list(community) for community in communities],
                "community_count": len(communities),
                "modularity": 0.5  # Mock modularity score
            }
        except Exception as e:
            return {"error": f"Community detection failed: {str(e)}"}

    async def _analyze_centrality(self, source_entity: str) -> Dict[str, Any]:
        """Analyze centrality measures for entity"""

        if source_entity not in self.transaction_graph:
            return {"error": "Entity not found in graph"}

        try:
            # Calculate various centrality measures
            degree_centrality = nx.degree_centrality(self.transaction_graph)
            betweenness_centrality = nx.betweenness_centrality(
                self.transaction_graph)

            # For clustering coefficient, we need undirected graph
            undirected_graph = self.transaction_graph.to_undirected()
            clustering_coeff = nx.clustering(undirected_graph)

            entity_node = source_entity

            return {
                "entity": source_entity,
                "degree_centrality": degree_centrality.get(entity_node, 0),
                "betweenness_centrality": betweenness_centrality.get(entity_node, 0),
                "clustering_coefficient": clustering_coeff.get(entity_node, 0),
                "degree": self.transaction_graph.degree(entity_node),
                "neighbors": list(self.transaction_graph.neighbors(entity_node))
            }
        except Exception as e:
            return {"error": f"Centrality analysis failed: {str(e)}"}

    async def get_entity_profile(self, entity_type: str, entity_id: str) -> Dict[str, Any]:
        """Get comprehensive profile of an entity"""

        redis_client = await self._get_redis_client()

        # Try cache first
        profile_key = f"{self.node_prefix}{entity_type}:{entity_id}"
        cached_profile = await redis_client.get(profile_key)

        if cached_profile:
            try:
                return json.loads(cached_profile)
            except json.JSONDecodeError:
                pass

        # Build profile from database and graph
        connected = await self.find_connected_entities(entity_type, entity_id)
        suspicious_patterns = await self.detect_suspicious_patterns(entity_type, entity_id)

        entity_node = f"{entity_type}:{entity_id}"

        profile = {
            "entity_type": entity_type,
            "entity_id": entity_id,
            "connected_entities": {k: len(v) for k, v in connected.items()},
            "suspicious_patterns": len(suspicious_patterns),
            "risk_indicators": suspicious_patterns,
            "graph_metrics": {
                "in_graph": entity_node in self.transaction_graph,
                "degree": self.transaction_graph.degree(entity_node) if entity_node in self.transaction_graph else 0
            },
            "last_updated": datetime.utcnow().isoformat()
        }

        # Cache profile
        # 30 minutes
        await redis_client.setex(profile_key, 1800, json.dumps(profile))

        return profile

    async def cleanup_old_relationships(self, days_threshold: int = 90) -> int:
        """Clean up old relationships to maintain graph performance"""
        # This would be implemented as a background task
        # For now, return 0 as placeholder
        return 0
