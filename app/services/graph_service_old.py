"""
Graph-based Entity Relationship Service
Tracks connections between users, devices, and merchants for indirect risk detection
"""
import json
import networkx as nx
from typing import Dict, List, Any, Set, Tuple, Optional
from datetime import datetime, timedelta
import redis
from collections import defaultdict

from app.models import GraphQueryRequest, GraphQueryResponse, TransactionPayload
from src.config import settings


class GraphService:
    """Manages entity relationship graphs for fraud detection"""

    def __init__(self):
        self.redis_client = redis.from_url(settings.redis_url)
        self.graph_prefix = "graph:"
        self.edge_prefix = "edges:"
        self.node_prefix = "nodes:"

    async def add_transaction_to_graph(self, transaction: TransactionPayload) -> None:
        """Add a transaction to the entity relationship graph"""
        timestamp = datetime.utcnow().isoformat()

        # Create edges for user-device, user-merchant, device-merchant relationships
        edges = []

        if transaction.device_fingerprint:
            edges.extend([
                ("user", transaction.user_id, "device",
                 transaction.device_fingerprint, "uses_device"),
                ("device", transaction.device_fingerprint,
                 "user", transaction.user_id, "used_by")
            ])

        if transaction.merchant_id:
            edges.extend([
                ("user", transaction.user_id, "merchant",
                 transaction.merchant_id, "transacts_with"),
                ("merchant", transaction.merchant_id, "user",
                 transaction.user_id, "receives_from")
            ])

            if transaction.device_fingerprint:
                edges.extend([
                    ("device", transaction.device_fingerprint,
                     "merchant", transaction.merchant_id, "used_at"),
                    ("merchant", transaction.merchant_id, "device",
                     transaction.device_fingerprint, "accessed_by")
                ])

        # Store edges with transaction metadata
        for edge in edges:
            await self._store_edge(edge, transaction, timestamp)

        # Update node information
        await self._update_node("user", transaction.user_id, transaction, timestamp)

        if transaction.device_fingerprint:
            await self._update_node("device", transaction.device_fingerprint, transaction, timestamp)

        if transaction.merchant_id:
            await self._update_node("merchant", transaction.merchant_id, transaction, timestamp)

    async def query_entity_relationships(self, request: GraphQueryRequest) -> GraphQueryResponse:
        """Query relationships for an entity"""

        # Get direct connections
        direct_connections = await self._get_direct_connections(
            request.entity_type,
            request.entity_id
        )

        # Calculate indirect risk score if requested
        indirect_risk_score = 0.0
        suspicious_patterns = []
        network_metrics = {}

        if request.include_risk_propagation:
            graph = await self._build_local_graph(
                request.entity_type,
                request.entity_id,
                request.relationship_depth
            )

            indirect_risk_score = await self._calculate_indirect_risk(graph, request.entity_id)
            suspicious_patterns = await self._detect_suspicious_patterns(graph, request.entity_id)
            network_metrics = await self._calculate_network_metrics(graph, request.entity_id)

        return GraphQueryResponse(
            entity_id=request.entity_id,
            direct_connections=direct_connections,
            indirect_risk_score=indirect_risk_score,
            suspicious_patterns=suspicious_patterns,
            network_metrics=network_metrics
        )

    async def get_connected_entities(
        self,
        entity_type: str,
        entity_id: str,
        max_depth: int = 2
    ) -> Dict[str, Set[str]]:
        """Get all entities connected within max_depth hops"""

        visited = set()
        to_visit = [(entity_type, entity_id, 0)]
        connections = defaultdict(set)

        while to_visit:
            current_type, current_id, depth = to_visit.pop(0)

            if depth >= max_depth or (current_type, current_id) in visited:
                continue

            visited.add((current_type, current_id))
            connections[current_type].add(current_id)

            # Get direct neighbors
            neighbors = await self._get_direct_connections(current_type, current_id)

            for neighbor in neighbors:
                neighbor_type = neighbor.get("entity_type")
                neighbor_id = neighbor.get("entity_id")

                if neighbor_type and neighbor_id:
                    to_visit.append((neighbor_type, neighbor_id, depth + 1))

        # Convert sets to lists for JSON serialization
        return {k: list(v) for k, v in connections.items()}

    async def detect_fraud_rings(self, entity_type: str, entity_id: str) -> List[Dict[str, Any]]:
        """Detect potential fraud rings involving the entity"""

        # Build extended graph around the entity
        graph = await self._build_local_graph(entity_type, entity_id, depth=3)

        fraud_rings = []

        # Look for dense clusters of entities
        try:
            # Find cliques (fully connected subgraphs)
            cliques = list(nx.find_cliques(graph.to_undirected()))

            for clique in cliques:
                if len(clique) >= 3:  # Minimum ring size
                    ring_score = await self._calculate_ring_risk_score(graph, clique)

                    if ring_score > 0.6:  # High-risk threshold
                        fraud_rings.append({
                            "ring_members": list(clique),
                            "risk_score": ring_score,
                            "connection_strength": self._calculate_connection_strength(graph, clique),
                            "temporal_clustering": await self._analyze_temporal_patterns(clique)
                        })

        except Exception as e:
            # NetworkX operations can fail on complex graphs
            print(f"Error detecting fraud rings: {e}")

        return fraud_rings

    async def _store_edge(
        self,
        edge: Tuple[str, str, str, str, str],
        transaction: TransactionPayload,
        timestamp: str
    ) -> None:
        """Store an edge in the graph"""
        source_type, source_id, target_type, target_id, relationship = edge

        edge_key = f"{self.edge_prefix}{source_type}:{source_id}:{target_type}:{target_id}"

        edge_data = {
            "source_type": source_type,
            "source_id": source_id,
            "target_type": target_type,
            "target_id": target_id,
            "relationship": relationship,
            "timestamp": timestamp,
            "transaction_id": transaction.transaction_id,
            "amount": transaction.amount,
            "payment_type": transaction.payment_type.value
        }

        # Store edge data
        self.redis_client.lpush(edge_key, json.dumps(edge_data))
        self.redis_client.ltrim(edge_key, 0, 999)  # Keep last 1000 edges
        self.redis_client.expire(edge_key, 90 * 24 * 3600)  # 90 days retention

        # Index for quick lookups
        source_index = f"{self.graph_prefix}from:{source_type}:{source_id}"
        target_index = f"{self.graph_prefix}to:{target_type}:{target_id}"

        self.redis_client.sadd(source_index, f"{target_type}:{target_id}")
        self.redis_client.sadd(target_index, f"{source_type}:{source_id}")

        self.redis_client.expire(source_index, 90 * 24 * 3600)
        self.redis_client.expire(target_index, 90 * 24 * 3600)

    async def _update_node(
        self,
        node_type: str,
        node_id: str,
        transaction: TransactionPayload,
        timestamp: str
    ) -> None:
        """Update node information"""
        node_key = f"{self.node_prefix}{node_type}:{node_id}"

        # Get existing node data
        existing_data = self.redis_client.get(node_key)

        if existing_data:
            node_data = json.loads(existing_data)
        else:
            node_data = {
                "node_type": node_type,
                "node_id": node_id,
                "first_seen": timestamp,
                "transaction_count": 0,
                "total_amount": 0.0,
                "risk_indicators": []
            }

        # Update with new transaction
        node_data["last_seen"] = timestamp
        node_data["transaction_count"] += 1
        node_data["total_amount"] += transaction.amount

        # Add risk indicators based on transaction patterns
        if transaction.amount > 10000:
            if "high_value_transactions" not in node_data["risk_indicators"]:
                node_data["risk_indicators"].append("high_value_transactions")

        # Store updated node data
        self.redis_client.setex(
            node_key,
            90 * 24 * 3600,  # 90 days
            json.dumps(node_data)
        )

    async def _get_direct_connections(
        self,
        entity_type: str,
        entity_id: str
    ) -> List[Dict[str, Any]]:
        """Get direct connections for an entity"""

        source_index = f"{self.graph_prefix}from:{entity_type}:{entity_id}"
        connected_entities = self.redis_client.smembers(source_index)

        connections = []

        for entity in connected_entities:
            entity_str = entity.decode(
                'utf-8') if isinstance(entity, bytes) else entity
            target_type, target_id = entity_str.split(':', 1)

            # Get edge details
            edge_key = f"{self.edge_prefix}{entity_type}:{entity_id}:{target_type}:{target_id}"
            edge_data = self.redis_client.lrange(
                edge_key, 0, 0)  # Get most recent edge

            if edge_data:
                edge_info = json.loads(edge_data[0])

                connections.append({
                    "entity_type": target_type,
                    "entity_id": target_id,
                    "relationship": edge_info.get("relationship"),
                    "last_interaction": edge_info.get("timestamp"),
                    "interaction_count": self.redis_client.llen(edge_key),
                    "total_amount": sum(
                        json.loads(data).get("amount", 0)
                        for data in self.redis_client.lrange(edge_key, 0, -1)
                    )
                })

        return connections

    async def _build_local_graph(
        self,
        entity_type: str,
        entity_id: str,
        depth: int = 2
    ) -> nx.DiGraph:
        """Build a NetworkX graph around an entity"""

        graph = nx.DiGraph()
        visited = set()
        to_visit = [(entity_type, entity_id, 0)]

        while to_visit:
            current_type, current_id, current_depth = to_visit.pop(0)

            if current_depth >= depth or (current_type, current_id) in visited:
                continue

            visited.add((current_type, current_id))
            node_label = f"{current_type}:{current_id}"

            # Add node to graph
            if not graph.has_node(node_label):
                node_data = await self._get_node_data(current_type, current_id)
                graph.add_node(node_label, **node_data)

            # Get and add edges
            connections = await self._get_direct_connections(current_type, current_id)

            for conn in connections:
                target_label = f"{conn['entity_type']}:{conn['entity_id']}"

                # Add edge with weight based on interaction strength
                weight = min(conn.get('interaction_count', 1) / 10.0, 1.0)
                graph.add_edge(
                    node_label,
                    target_label,
                    weight=weight,
                    relationship=conn.get('relationship'),
                    amount=conn.get('total_amount', 0)
                )

                # Add to visit queue
                to_visit.append(
                    (conn['entity_type'], conn['entity_id'], current_depth + 1))

        return graph

    async def _get_node_data(self, node_type: str, node_id: str) -> Dict[str, Any]:
        """Get node data from storage"""
        node_key = f"{self.node_prefix}{node_type}:{node_id}"
        node_data = self.redis_client.get(node_key)

        if node_data:
            return json.loads(node_data)

        return {
            "node_type": node_type,
            "node_id": node_id,
            "risk_score": 0.5  # Default neutral risk
        }

    async def _calculate_indirect_risk(self, graph: nx.DiGraph, entity_id: str) -> float:
        """Calculate indirect risk score based on connected entities"""

        entity_nodes = [n for n in graph.nodes() if entity_id in n]
        if not entity_nodes:
            return 0.0

        entity_node = entity_nodes[0]
        total_risk = 0.0
        risk_contributions = 0

        # Get neighbors and their risk scores
        for neighbor in graph.neighbors(entity_node):
            neighbor_data = graph.nodes[neighbor]
            neighbor_risk = neighbor_data.get('risk_score', 0.5)

            # Weight by connection strength
            edge_data = graph.edges[entity_node, neighbor]
            connection_weight = edge_data.get('weight', 0.1)

            total_risk += neighbor_risk * connection_weight
            risk_contributions += connection_weight

        if risk_contributions == 0:
            return 0.0

        return min(total_risk / risk_contributions, 1.0)

    async def _detect_suspicious_patterns(self, graph: nx.DiGraph, entity_id: str) -> List[str]:
        """Detect suspicious patterns in the entity's network"""

        patterns = []
        entity_nodes = [n for n in graph.nodes() if entity_id in n]

        if not entity_nodes:
            return patterns

        entity_node = entity_nodes[0]
        neighbors = list(graph.neighbors(entity_node))

        # Pattern 1: Many connections to the same device
        device_connections = [n for n in neighbors if n.startswith('device:')]
        if len(device_connections) > 5:
            patterns.append("multiple_device_usage")

        # Pattern 2: Rapid connections to new entities
        recent_connections = []
        for neighbor in neighbors:
            edge_data = graph.edges[entity_node, neighbor]
            # This would need timestamp analysis in a real implementation

        # Pattern 3: High-value transactions in short time
        high_value_neighbors = []
        for neighbor in neighbors:
            edge_data = graph.edges[entity_node, neighbor]
            if edge_data.get('amount', 0) > 5000:
                high_value_neighbors.append(neighbor)

        if len(high_value_neighbors) > 3:
            patterns.append("frequent_high_value_transactions")

        # Pattern 4: Connection to known risky entities
        for neighbor in neighbors:
            neighbor_data = graph.nodes[neighbor]
            if neighbor_data.get('risk_score', 0) > 0.8:
                patterns.append("connected_to_high_risk_entity")
                break

        return patterns

    async def _calculate_network_metrics(self, graph: nx.DiGraph, entity_id: str) -> Dict[str, float]:
        """Calculate network-based metrics for the entity"""

        entity_nodes = [n for n in graph.nodes() if entity_id in n]
        if not entity_nodes:
            return {}

        entity_node = entity_nodes[0]

        try:
            metrics = {
                "degree_centrality": nx.degree_centrality(graph).get(entity_node, 0),
                "betweenness_centrality": nx.betweenness_centrality(graph).get(entity_node, 0),
                "clustering_coefficient": nx.clustering(graph.to_undirected()).get(entity_node, 0),
                "neighbor_count": len(list(graph.neighbors(entity_node)))
            }

            # Pagerank centrality (if graph is large enough)
            if len(graph.nodes()) > 1:
                pagerank = nx.pagerank(graph)
                metrics["pagerank_centrality"] = pagerank.get(entity_node, 0)

            return metrics

        except Exception as e:
            print(f"Error calculating network metrics: {e}")
            return {
                "neighbor_count": len(list(graph.neighbors(entity_node)))
            }

    async def _calculate_ring_risk_score(self, graph: nx.DiGraph, clique: List[str]) -> float:
        """Calculate risk score for a potential fraud ring"""

        # Factors that increase fraud ring probability:
        # 1. High transaction volumes between members
        # 2. Temporal clustering of transactions
        # 3. Unusual transaction patterns
        # 4. New entities (accounts/devices)

        total_score = 0.0
        factors = 0

        for i, node1 in enumerate(clique):
            for node2 in clique[i+1:]:
                if graph.has_edge(node1, node2):
                    edge_data = graph.edges[node1, node2]

                    # High transaction volume
                    amount = edge_data.get('amount', 0)
                    if amount > 10000:
                        total_score += 0.3
                        factors += 1

                    # Connection strength
                    weight = edge_data.get('weight', 0)
                    if weight > 0.5:
                        total_score += 0.2
                        factors += 1

        # Check for new entities in the ring
        new_entity_count = 0
        for node in clique:
            node_data = graph.nodes[node]
            # This would need actual timestamp analysis
            # For now, using transaction count as proxy
            if node_data.get('transaction_count', 0) < 10:
                new_entity_count += 1

        if new_entity_count >= len(clique) * 0.5:  # More than half are new
            total_score += 0.4
            factors += 1

        return min(total_score / max(factors, 1), 1.0) if factors > 0 else 0.0

    def _calculate_connection_strength(self, graph: nx.DiGraph, clique: List[str]) -> float:
        """Calculate the strength of connections within a clique"""

        total_weight = 0.0
        connection_count = 0

        for i, node1 in enumerate(clique):
            for node2 in clique[i+1:]:
                if graph.has_edge(node1, node2):
                    edge_data = graph.edges[node1, node2]
                    total_weight += edge_data.get('weight', 0)
                    connection_count += 1

                if graph.has_edge(node2, node1):
                    edge_data = graph.edges[node2, node1]
                    total_weight += edge_data.get('weight', 0)
                    connection_count += 1

        return total_weight / max(connection_count, 1)

    async def _analyze_temporal_patterns(self, clique: List[str]) -> Dict[str, Any]:
        """Analyze temporal patterns within a fraud ring"""

        # This would analyze transaction timestamps to detect:
        # - Burst patterns (many transactions in short time)
        # - Coordinated activity
        # - Unusual timing

        # Placeholder implementation
        return {
            "has_burst_activity": False,
            "coordinated_transactions": False,
            "unusual_timing": False
        }


# Global graph service instance
graph_service = GraphService()
