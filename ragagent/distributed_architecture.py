"""
Distributed architecture for scalable RAG processing
"""

import asyncio
import logging
import json
import time
import uuid
import os
from typing import Dict, List, Any, Optional, Tuple, Callable
from dataclasses import dataclass, asdict
from pathlib import Path
from datetime import datetime, timedelta
import concurrent.futures
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import multiprocessing as mp
from queue import Queue
import threading
import redis
import pickle
from enum import Enum

from .models import ProcessingRequest, ProcessingResponse, APIResponse
from .config import get_config
from .utils import CacheManager
from .pipeline import AgenticRAGPipeline

logger = logging.getLogger(__name__)


class NodeStatus(Enum):
    """Node status enumeration"""

    ACTIVE = "active"
    IDLE = "idle"
    BUSY = "busy"
    OFFLINE = "offline"
    MAINTENANCE = "maintenance"


class TaskPriority(Enum):
    """Task priority levels"""

    LOW = 1
    NORMAL = 2
    HIGH = 3
    CRITICAL = 4


@dataclass
class NodeInfo:
    """Information about a compute node"""

    node_id: str
    host: str
    port: int
    status: NodeStatus
    cpu_cores: int
    memory_gb: float
    gpu_available: bool
    gpu_memory_gb: float
    current_tasks: int
    max_tasks: int
    last_heartbeat: datetime
    capabilities: List[str]
    load_average: float = 0.0
    uptime: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "NodeInfo":
        """Create from dictionary"""
        data["last_heartbeat"] = datetime.fromisoformat(data["last_heartbeat"])
        return cls(**data)


@dataclass
class DistributedTask:
    """Distributed task representation"""

    task_id: str
    request_id: str
    task_type: str
    priority: TaskPriority
    payload: Dict[str, Any]
    created_at: datetime
    scheduled_at: Optional[datetime] = None
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    assigned_node: Optional[str] = None
    status: str = "pending"
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    retry_count: int = 0
    max_retries: int = 3

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        data = asdict(self)
        data["created_at"] = self.created_at.isoformat()
        if self.scheduled_at:
            data["scheduled_at"] = self.scheduled_at.isoformat()
        if self.started_at:
            data["started_at"] = self.started_at.isoformat()
        if self.completed_at:
            data["completed_at"] = self.completed_at.isoformat()
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DistributedTask":
        """Create from dictionary"""
        data["created_at"] = datetime.fromisoformat(data["created_at"])
        if data["scheduled_at"]:
            data["scheduled_at"] = datetime.fromisoformat(data["scheduled_at"])
        if data["started_at"]:
            data["started_at"] = datetime.fromisoformat(data["started_at"])
        if data["completed_at"]:
            data["completed_at"] = datetime.fromisoformat(data["completed_at"])
        return cls(**data)


class TaskScheduler:
    """Distributed task scheduler"""

    def __init__(self, redis_client: redis.Redis, config: Dict[str, Any] = None):
        self.redis = redis_client
        self.config = config or get_config().dict()
        self.logger = logging.getLogger(self.__class__.__name__)

        # Scheduler configuration
        self.scheduler_interval = self.config.get("scheduler_interval", 5)
        self.max_task_queue_size = self.config.get("max_task_queue_size", 1000)
        self.task_timeout = self.config.get("task_timeout", 300)
        self.heartbeat_interval = self.config.get("heartbeat_interval", 30)

        # Task queues by priority
        self.task_queues = {
            TaskPriority.LOW: Queue(maxsize=self.max_task_queue_size),
            TaskPriority.NORMAL: Queue(maxsize=self.max_task_queue_size),
            TaskPriority.HIGH: Queue(maxsize=self.max_task_queue_size),
            TaskPriority.CRITICAL: Queue(maxsize=self.max_task_queue_size),
        }

        # Node management
        self.nodes: Dict[str, NodeInfo] = {}
        self.node_lock = threading.Lock()

        # Task tracking
        self.active_tasks: Dict[str, DistributedTask] = {}
        self.completed_tasks: Dict[str, DistributedTask] = {}

        # Start scheduler threads
        self.running = True
        self.scheduler_thread = threading.Thread(target=self._scheduler_loop)
        self.heartbeat_thread = threading.Thread(target=self._heartbeat_loop)
        self.scheduler_thread.start()
        self.heartbeat_thread.start()

        self.logger.info("Task scheduler initialized")

    def _scheduler_loop(self):
        """Main scheduler loop"""
        while self.running:
            try:
                self._schedule_tasks()
                time.sleep(self.scheduler_interval)
            except Exception as e:
                self.logger.error(f"Error in scheduler loop: {e}")
                time.sleep(self.scheduler_interval)

    def _heartbeat_loop(self):
        """Heartbeat monitoring loop"""
        while self.running:
            try:
                self._monitor_nodes()
                time.sleep(self.heartbeat_interval)
            except Exception as e:
                self.logger.error(f"Error in heartbeat loop: {e}")
                time.sleep(self.heartbeat_interval)

    def _schedule_tasks(self):
        """Schedule tasks to available nodes"""
        with self.node_lock:
            available_nodes = [
                node
                for node in self.nodes.values()
                if node.status == NodeStatus.IDLE
                and node.current_tasks < node.max_tasks
            ]

        if not available_nodes:
            return

        # Schedule tasks by priority
        for priority in [
            TaskPriority.CRITICAL,
            TaskPriority.HIGH,
            TaskPriority.NORMAL,
            TaskPriority.LOW,
        ]:
            if self.task_queues[priority].empty():
                continue

            # Get best node for this task
            best_node = self._select_best_node(available_nodes, priority)
            if not best_node:
                continue

            # Get task from queue
            try:
                task = self.task_queues[priority].get_nowait()
                task.assigned_node = best_node.node_id
                task.scheduled_at = datetime.now()
                task.status = "scheduled"

                # Store task in Redis
                self.redis.set(f"task:{task.task_id}", pickle.dumps(task))
                self.redis.lpush(f"node:{best_node.node_id}:tasks", task.task_id)

                # Update node status
                best_node.current_tasks += 1
                best_node.status = NodeStatus.BUSY

                # Update node in Redis
                self.redis.set(f"node:{best_node.node_id}", pickle.dumps(best_node))

                self.logger.info(
                    f"Scheduled task {task.task_id} to node {best_node.node_id}"
                )

            except Exception as e:
                self.logger.error(f"Error scheduling task: {e}")

    def _select_best_node(
        self, available_nodes: List[NodeInfo], priority: TaskPriority
    ) -> Optional[NodeInfo]:
        """Select best node for task based on various factors"""
        if not available_nodes:
            return None

        # Score nodes based on load, capabilities, and priority
        scored_nodes = []

        for node in available_nodes:
            score = 0.0

            # Load factor (lower is better)
            load_factor = node.current_tasks / node.max_tasks
            score += (1.0 - load_factor) * 0.4

            # Capability matching
            if priority == TaskPriority.CRITICAL and node.gpu_available:
                score += 0.3
            elif priority == TaskPriority.HIGH and node.cpu_cores >= 4:
                score += 0.2

            # Load average
            score += (1.0 - node.load_average) * 0.3

            scored_nodes.append((node, score))

        # Return node with highest score
        return max(scored_nodes, key=lambda x: x[1])[0]

    def _monitor_nodes(self):
        """Monitor node health and status"""
        current_time = datetime.now()

        with self.node_lock:
            for node_id, node in list(self.nodes.items()):
                # Check if node is offline (no heartbeat for 3 intervals)
                if (
                    current_time - node.last_heartbeat
                ).total_seconds() > self.heartbeat_interval * 3:
                    node.status = NodeStatus.OFFLINE
                    self.logger.warning(f"Node {node_id} marked as offline")

                    # Reassign tasks from offline node
                    self._reassign_tasks_from_node(node_id)

                # Update node in Redis
                self.redis.set(f"node:{node_id}", pickle.dumps(node))

    def _reassign_tasks_from_node(self, node_id: str):
        """Reassign tasks from failed/offline node"""
        # Get tasks assigned to this node
        task_ids = self.redis.lrange(f"node:{node_id}:tasks", 0, -1)

        for task_id_bytes in task_ids:
            task_id = task_id_bytes.decode("utf-8")

            # Get task from Redis
            task_data = self.redis.get(f"task:{task_id}")
            if task_data:
                task = pickle.loads(task_data)

                # Requeue task if it's still pending
                if task.status in ["pending", "scheduled"]:
                    task.assigned_node = None
                    task.status = "pending"
                    task.retry_count += 1

                    # Add back to appropriate queue
                    self.task_queues[task.priority].put(task)

                    self.logger.info(f"Reassigned task {task_id} from node {node_id}")

    def submit_task(self, task: DistributedTask) -> bool:
        """Submit a task to the scheduler"""
        if task.priority not in self.task_queues:
            return False

        try:
            self.task_queues[task.priority].put(task, timeout=1)
            self.logger.info(
                f"Submitted task {task.task_id} with priority {task.priority.name}"
            )
            return True
        except Exception as e:
            self.logger.error(f"Error submitting task: {e}")
            return False

    def register_node(self, node_info: NodeInfo) -> bool:
        """Register a new compute node"""
        with self.node_lock:
            self.nodes[node_info.node_id] = node_info
            self.logger.info(f"Registered node {node_info.node_id}")
            return True

    def unregister_node(self, node_id: str) -> bool:
        """Unregister a compute node"""
        with self.node_lock:
            if node_id in self.nodes:
                del self.nodes[node_id]
                self.logger.info(f"Unregistered node {node_id}")
                return True
            return False

    def get_node_info(self, node_id: str) -> Optional[NodeInfo]:
        """Get information about a specific node"""
        with self.node_lock:
            return self.nodes.get(node_id)

    def get_all_nodes(self) -> List[NodeInfo]:
        """Get information about all nodes"""
        with self.node_lock:
            return list(self.nodes.values())

    def get_task_status(self, task_id: str) -> Optional[str]:
        """Get status of a specific task"""
        # Check active tasks
        if task_id in self.active_tasks:
            return self.active_tasks[task_id].status

        # Check Redis
        task_data = self.redis.get(f"task:{task_id}")
        if task_data:
            task = pickle.loads(task_data)
            return task.status

        return None

    def shutdown(self):
        """Shutdown the scheduler"""
        self.running = False
        self.scheduler_thread.join()
        self.heartbeat_thread.join()
        self.logger.info("Task scheduler shutdown")


class ComputeNode:
    """Compute node for distributed processing"""

    def __init__(
        self, node_id: str, host: str, port: int, config: Dict[str, Any] = None
    ):
        self.node_id = node_id
        self.host = host
        self.port = port
        self.config = config or get_config().dict()
        self.logger = logging.getLogger(self.__class__.__name__)

        # Node capabilities
        self.cpu_cores = mp.cpu_count()
        self.memory_gb = self._get_memory_gb()
        self.gpu_available = self._check_gpu()
        self.gpu_memory_gb = self._get_gpu_memory_gb()

        # Node state
        self.status = NodeStatus.IDLE
        self.current_tasks = 0
        self.max_tasks = self.config.get("max_tasks_per_node", 4)
        self.last_heartbeat = datetime.now()
        self.uptime = 0.0
        self.start_time = datetime.now()

        # Task executor
        self.executor = ThreadPoolExecutor(max_workers=self.max_tasks)
        self.task_futures: Dict[str, concurrent.futures.Future] = {}

        # Redis client
        self.redis = redis.Redis(
            host=self.config.get("redis_host", "localhost"),
            port=self.config.get("redis_port", 6379),
        )

        # Start node monitoring
        self.running = True
        self.monitor_thread = threading.Thread(target=self._monitor_loop)
        self.monitor_thread.start()

        self.logger.info(f"Compute node {node_id} initialized")

    def _get_memory_gb(self) -> float:
        """Get available memory in GB"""
        try:
            import psutil

            return psutil.virtual_memory().available / (1024**3)
        except ImportError:
            return 8.0  # Default estimate

    def _check_gpu(self) -> bool:
        """Check if GPU is available"""
        try:
            import torch

            return torch.cuda.is_available()
        except ImportError:
            return False

    def _get_gpu_memory_gb(self) -> float:
        """Get GPU memory in GB"""
        try:
            import torch

            if torch.cuda.is_available():
                return torch.cuda.get_device_properties(0).total_memory / (1024**3)
        except ImportError:
            pass
        return 0.0

    def _monitor_loop(self):
        """Node monitoring loop"""
        while self.running:
            try:
                self._update_status()
                self._check_task_completion()
                self._send_heartbeat()
                time.sleep(5)
            except Exception as e:
                self.logger.error(f"Error in monitor loop: {e}")
                time.sleep(5)

    def _update_status(self):
        """Update node status"""
        self.uptime = (datetime.now() - self.start_time).total_seconds()

        # Calculate load average
        if hasattr(os, "getloadavg"):
            self.load_average = os.getloadavg()[0]
        else:
            self.load_average = (
                sum(future.done() for future in self.task_futures.values())
                / len(self.task_futures)
                if self.task_futures
                else 0.0
            )

        # Update status based on current tasks
        if self.current_tasks >= self.max_tasks:
            self.status = NodeStatus.BUSY
        elif self.current_tasks > 0:
            self.status = NodeStatus.ACTIVE
        else:
            self.status = NodeStatus.IDLE

    def _check_task_completion(self):
        """Check for completed tasks"""
        completed_tasks = []

        for task_id, future in self.task_futures.items():
            if future.done():
                completed_tasks.append(task_id)

                try:
                    result = future.result()
                    self._handle_task_completion(task_id, result, None)
                except Exception as e:
                    self._handle_task_completion(task_id, None, str(e))

        # Remove completed tasks
        for task_id in completed_tasks:
            del self.task_futures[task_id]

    def _handle_task_completion(self, task_id: str, result: Any, error: str):
        """Handle task completion"""
        self.current_tasks -= 1

        # Update task in Redis
        task_data = self.redis.get(f"task:{task_id}")
        if task_data:
            task = pickle.loads(task_data)
            task.completed_at = datetime.now()
            task.status = "completed" if error is None else "failed"
            task.result = result if error is None else None
            task.error = error

            self.redis.set(f"task:{task_id}", pickle.dumps(task))

            # Store completed task
            self.redis.lpush("completed_tasks", task_id)

        self.logger.info(f"Task {task_id} completed with error: {error}")

    def _send_heartbeat(self):
        """Send heartbeat to scheduler"""
        node_info = NodeInfo(
            node_id=self.node_id,
            host=self.host,
            port=self.port,
            status=self.status,
            cpu_cores=self.cpu_cores,
            memory_gb=self.memory_gb,
            gpu_available=self.gpu_available,
            gpu_memory_gb=self.gpu_memory_gb,
            current_tasks=self.current_tasks,
            max_tasks=self.max_tasks,
            last_heartbeat=datetime.now(),
            capabilities=["rag_processing", "document_parsing", "ai_inference"],
        )

        self.redis.set(f"node:{self.node_id}", pickle.dumps(node_info))
        self.last_heartbeat = datetime.now()

    def execute_task(self, task: DistributedTask) -> bool:
        """Execute a task on this node"""
        if self.current_tasks >= self.max_tasks:
            return False

        try:
            # Submit task to executor
            future = self.executor.submit(self._process_task, task)
            self.task_futures[task.task_id] = future
            self.current_tasks += 1

            self.logger.info(f"Executing task {task.task_id} on node {self.node_id}")
            return True

        except Exception as e:
            self.logger.error(f"Error executing task: {e}")
            return False

    def _process_task(self, task: DistributedTask) -> Dict[str, Any]:
        """Process a task"""
        try:
            # Import required modules
            from .pipeline import AgenticRAGPipeline
            from .models import ProcessingRequest

            # Process based on task type
            if task.task_type == "rag_processing":
                return self._process_rag_task(task)
            elif task.task_type == "document_parsing":
                return self._process_parsing_task(task)
            else:
                raise ValueError(f"Unknown task type: {task.task_type}")

        except Exception as e:
            self.logger.error(f"Error processing task: {e}")
            raise

    def _process_rag_task(self, task: DistributedTask) -> Dict[str, Any]:
        """Process RAG task"""
        # Extract request from payload
        request_data = task.payload.get("request")
        request = ProcessingRequest(**request_data)

        # Create pipeline and process
        pipeline = AgenticRAGPipeline()
        result = pipeline.process_request(request)

        return {
            "result": result.dict() if result.success else None,
            "error": result.error.message if not result.success else None,
        }

    def _process_parsing_task(self, task: DistributedTask) -> Dict[str, Any]:
        """Process document parsing task"""
        from .parsers import ParserRegistry

        # Extract parsing parameters
        file_path = task.payload.get("file_path")
        document_type = task.payload.get("document_type")

        # Parse document
        parser_registry = ParserRegistry()
        chunks = parser_registry.parse_file(file_path)

        return {
            "chunks": [chunk.dict() for chunk in chunks],
            "chunk_count": len(chunks),
        }

    def get_status(self) -> Dict[str, Any]:
        """Get node status"""
        return {
            "node_id": self.node_id,
            "host": self.host,
            "port": self.port,
            "status": self.status.value,
            "cpu_cores": self.cpu_cores,
            "memory_gb": self.memory_gb,
            "gpu_available": self.gpu_available,
            "gpu_memory_gb": self.gpu_memory_gb,
            "current_tasks": self.current_tasks,
            "max_tasks": self.max_tasks,
            "load_average": self.load_average,
            "uptime": self.uptime,
            "last_heartbeat": self.last_heartbeat.isoformat(),
        }

    def shutdown(self):
        """Shutdown the compute node"""
        self.running = False

        # Cancel all running tasks
        for future in self.task_futures.values():
            future.cancel()

        # Wait for tasks to complete
        self.executor.shutdown(wait=True)

        # Send final heartbeat
        self.status = NodeStatus.OFFLINE
        self._send_heartbeat()

        self.monitor_thread.join()
        self.logger.info(f"Compute node {self.node_id} shutdown")


class DistributedRAGSystem:
    """Main distributed RAG system"""

    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or get_config().dict()
        self.logger = logging.getLogger(self.__class__.__name__)

        # Initialize Redis client
        self.redis = redis.Redis(
            host=self.config.get("redis_host", "localhost"),
            port=self.config.get("redis_port", 6379),
        )

        # Initialize components
        self.scheduler = TaskScheduler(self.redis, config)
        self.compute_nodes: Dict[str, ComputeNode] = {}

        # System metrics
        self.metrics = {
            "total_tasks": 0,
            "completed_tasks": 0,
            "failed_tasks": 0,
            "average_processing_time": 0.0,
            "system_throughput": 0.0,
        }

        # Start system monitoring
        self.running = True
        self.monitor_thread = threading.Thread(target=self._system_monitor_loop)
        self.monitor_thread.start()

        self.logger.info("Distributed RAG system initialized")

    def _system_monitor_loop(self):
        """System monitoring loop"""
        while self.running:
            try:
                self._update_metrics()
                time.sleep(60)  # Update metrics every minute
            except Exception as e:
                self.logger.error(f"Error in system monitor: {e}")
                time.sleep(60)

    def _update_metrics(self):
        """Update system metrics"""
        # Get completed tasks count
        completed_count = self.redis.llen("completed_tasks")
        self.metrics["completed_tasks"] = completed_count

        # Get system throughput
        current_time = time.time()
        recent_tasks = self.redis.lrange("completed_tasks", 0, -1)

        # Calculate throughput (tasks per minute)
        if recent_tasks:
            oldest_task_time = self.redis.get(
                f"task:{recent_tasks[-1].decode('utf-8')}:completed_at"
            )
            if oldest_task_time:
                time_diff = current_time - float(oldest_task_time)
                if time_diff > 0:
                    self.metrics["system_throughput"] = len(recent_tasks) / (
                        time_diff / 60
                    )

        # Store metrics
        self.redis.set("system_metrics", json.dumps(self.metrics))

    def add_compute_node(self, node: ComputeNode) -> bool:
        """Add a compute node to the system"""
        self.compute_nodes[node.node_id] = node

        # Register with scheduler
        node_info = NodeInfo(
            node_id=node.node_id,
            host=node.host,
            port=node.port,
            status=node.status,
            cpu_cores=node.cpu_cores,
            memory_gb=node.memory_gb,
            gpu_available=node.gpu_available,
            gpu_memory_gb=node.gpu_memory_gb,
            current_tasks=node.current_tasks,
            max_tasks=node.max_tasks,
            last_heartbeat=node.last_heartbeat,
            capabilities=node.get_status().get("capabilities", []),
        )

        self.scheduler.register_node(node_info)

        self.logger.info(f"Added compute node {node.node_id}")
        return True

    def remove_compute_node(self, node_id: str) -> bool:
        """Remove a compute node from the system"""
        if node_id in self.compute_nodes:
            node = self.compute_nodes[node_id]
            node.shutdown()
            del self.compute_nodes[node_id]

            # Unregister from scheduler
            self.scheduler.unregister_node(node_id)

            self.logger.info(f"Removed compute node {node_id}")
            return True
        return False

    def process_request(self, request: ProcessingRequest) -> APIResponse:
        """Process a request using distributed architecture"""
        start_time = time.time()

        try:
            # Create distributed task
            task = DistributedTask(
                task_id=str(uuid.uuid4()),
                request_id=str(uuid.uuid4()),
                task_type="rag_processing",
                priority=TaskPriority.NORMAL,
                payload={"request": request.dict()},
                created_at=datetime.now(),
            )

            # Submit task to scheduler
            if self.scheduler.submit_task(task):
                self.metrics["total_tasks"] += 1

                # Wait for task completion (with timeout)
                timeout = self.config.get("task_timeout", 300)
                start_wait = time.time()

                while time.time() - start_wait < timeout:
                    status = self.scheduler.get_task_status(task.task_id)
                    if status == "completed":
                        # Get result
                        task_data = self.redis.get(f"task:{task.task_id}")
                        if task_data:
                            completed_task = pickle.loads(task_data)
                            if completed_task.result:
                                return APIResponse(
                                    success=True, data=completed_task.result
                                )

                    elif status == "failed":
                        # Get error
                        task_data = self.redis.get(f"task:{task.task_id}")
                        if task_data:
                            failed_task = pickle.loads(task_data)
                            return APIResponse(success=False, error=failed_task.error)

                    time.sleep(1)

                # Timeout
                return APIResponse(
                    success=False,
                    error={
                        "error": "Task timeout",
                        "message": "Task processing timed out",
                    },
                )

            else:
                return APIResponse(
                    success=False,
                    error={
                        "error": "SchedulerError",
                        "message": "Failed to submit task to scheduler",
                    },
                )

        except Exception as e:
            self.logger.error(f"Error processing request: {e}")
            return APIResponse(
                success=False, error={"error": "ProcessingError", "message": str(e)}
            )

    def get_system_status(self) -> Dict[str, Any]:
        """Get system status"""
        nodes_status = [node.get_status() for node in self.compute_nodes.values()]

        return {
            "system_metrics": self.metrics,
            "nodes": nodes_status,
            "scheduler_info": {
                "total_tasks": sum(
                    queue.qsize() for queue in self.scheduler.task_queues.values()
                ),
                "active_tasks": len(self.scheduler.active_tasks),
                "registered_nodes": len(self.scheduler.nodes),
            },
        }

    def shutdown(self):
        """Shutdown the distributed system"""
        self.running = False

        # Shutdown all compute nodes
        for node in self.compute_nodes.values():
            node.shutdown()

        # Shutdown scheduler
        self.scheduler.shutdown()

        # Wait for monitoring thread
        self.monitor_thread.join()

        self.logger.info("Distributed RAG system shutdown")
