"""
Graph Service for Entity Relationship Analysis
Manages graph-based fraud detection using NetworkX and Redis
"""
from typing import Dict, List, Any, Optional
from datetime import datetime
import aioredis
import networkx as nx
import json

from app.models import GraphQueryRequest, GraphQueryResponse
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
            self.db_service.create_entity_relationship(
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
            self.db_service.create_entity_relationship(
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
            self.db_service.create_entity_relationship(
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
    ) -> Dict[str, Any]:
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
            # Use getattr to safely access SQLAlchemy model attributes
            source_type = getattr(rel, 'source_type', '')
            source_id = getattr(rel, 'source_id', '')
            target_type = getattr(rel, 'target_type', '')
            target_id = getattr(rel, 'target_id', '')

            # Determine if this entity is source or target
            if source_type == entity_type and source_id == entity_id:
                # This entity is the source
                target_key = f"{target_type}s"
                if target_key not in connections:
                    connections[target_key] = set()
                connections[target_key].add(target_id)
            elif target_type == entity_type and target_id == entity_id:
                # This entity is the target
                source_key = f"{source_type}s"
                if source_key not in connections:
                    connections[source_key] = set()
                connections[source_key].add(source_id)

        # Cache result
        cacheable_connections = {k: list(v) for k, v in connections.items()}
        await redis_client.setex(
            cache_key, 3600, json.dumps(cacheable_connections))  # 1 hour cache

        # Convert to expected format for GraphQueryResponse
        direct_connections = []
        for rel_type, entity_ids in connections.items():
            for eid in entity_ids:
                direct_connections.append({
                    "entity_id": eid,
                    "entity_type": rel_type.rstrip('s'),
                    "relationship_type": "transaction"
                })

        return {
            "connections": direct_connections,
            "risk_score": 0.3,  # Mock risk score
            "patterns": [],
            "metrics": {"connection_count": len(direct_connections)}
        }

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

    async def query_graph(self, query: GraphQueryRequest) -> Dict[str, Any]:
        """Execute graph query and return results"""

        if query.query_type == "shortest_path":
            return await self._find_shortest_path(
                query.entity_id, query.target_entity_id or query.entity_id, query.max_depth or 5)

        elif query.query_type == "connected_components":
            return await self._find_connected_components(
                query.entity_id, query.max_depth or 3)

        elif query.query_type == "community_detection":
            return await self._detect_communities(query.max_depth or 3)

        elif query.query_type == "centrality_analysis":
            return await self._analyze_centrality(query.entity_id)

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

            # Safe degree access
            node_degree = 0
            if entity_node in self.transaction_graph:
                node_degree = len(
                    list(self.transaction_graph.neighbors(entity_node)))

            return {
                "entity": source_entity,
                "degree_centrality": degree_centrality.get(entity_node, 0),
                "betweenness_centrality": betweenness_centrality.get(entity_node, 0),
                "clustering_coefficient": clustering_coeff.get(entity_node, 0) if isinstance(clustering_coeff, dict) else 0,
                "degree": node_degree,
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

        # Safe degree calculation
        node_degree = 0
        if entity_node in self.transaction_graph:
            node_degree = len(
                list(self.transaction_graph.neighbors(entity_node)))

        profile = {
            "entity_type": entity_type,
            "entity_id": entity_id,
            "connected_entities": {k: len(v) for k, v in connected.items()},
            "suspicious_patterns": len(suspicious_patterns),
            "risk_indicators": suspicious_patterns,
            "graph_metrics": {
                "in_graph": entity_node in self.transaction_graph,
                "degree": node_degree
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

    async def query_entity_relationships(self, request: 'GraphQueryRequest') -> 'GraphQueryResponse':
        """Query entity relationships based on request parameters"""
        from app.models import GraphQueryResponse

        # Extract query parameters
        entity_id = request.entity_id
        entity_type = request.entity_type
        query_type = request.query_type
        max_depth = request.max_depth

        if query_type == "shortest_path":
            result = await self._find_shortest_path(
                entity_id,
                request.target_entity_id or entity_id,
                max_depth
            )
        elif query_type == "connected_components":
            result = await self._find_connected_components(entity_id, max_depth)
        elif query_type == "community_detection":
            result = await self._detect_communities(max_depth)
        elif query_type == "centrality_analysis":
            result = await self._analyze_centrality(entity_id)
        else:
            # Default: find connected entities
            result = await self.find_connected_entities(entity_type, entity_id, max_depth)

        return GraphQueryResponse(
            entity_id=entity_id,
            direct_connections=result.get(
                "connections", []) if isinstance(result, dict) else [],
            indirect_risk_score=float(result.get(
                "risk_score", 0.0)) if isinstance(result, dict) else 0.0,
            suspicious_patterns=result.get(
                "patterns", []) if isinstance(result, dict) else [],
            network_metrics=result.get(
                "metrics", {}) if isinstance(result, dict) else {}
        )

    async def detect_fraud_rings(self, entity_type: str, entity_id: str) -> List[Dict[str, Any]]:
        """Detect potential fraud rings involving an entity"""

        # Get connected entities up to depth 3
        connected = await self.find_connected_entities(entity_type, entity_id, max_depth=3)

        # Analyze suspicious patterns
        suspicious_patterns = await self.detect_suspicious_patterns(entity_type, entity_id)

        fraud_rings = []

        # Look for clusters of highly connected entities with suspicious patterns
        for pattern in suspicious_patterns:
            if isinstance(pattern, dict) and pattern.get("risk_level") in ["HIGH", "MEDIUM"]:
                ring = {
                    "ring_id": f"ring_{entity_id}_{len(fraud_rings)}",
                    "center_entity": {"type": entity_type, "id": entity_id},
                    "ring_members": pattern.get("entities_involved", []),
                    "risk_score": pattern.get("confidence", 0.5),
                    "suspicious_indicators": [pattern.get("pattern_type", "unknown")],
                    "relationship_strength": pattern.get("confidence", 0.5),
                    "detected_at": datetime.utcnow().isoformat()
                }
                fraud_rings.append(ring)

        # Also check for dense subgraphs in connected entities
        all_connected_entities = []
        if isinstance(connected, dict) and connected.get("connections"):
            all_connected_entities = connected.get("connections", [])

        if len(all_connected_entities) > 3:
            # Simple heuristic: if many entities are connected to each other
            ring = {
                "ring_id": f"dense_ring_{entity_id}",
                "center_entity": {"type": entity_type, "id": entity_id},
                "ring_members": [conn.get("entity_id", "") for conn in all_connected_entities[:10]],
                "risk_score": min(0.8, len(all_connected_entities) * 0.1),
                "suspicious_indicators": ["high_connectivity", "multiple_shared_relationships"],
                "relationship_strength": 0.5,
                "detected_at": datetime.utcnow().isoformat()
            }
            fraud_rings.append(ring)

        return fraud_rings
