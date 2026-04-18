import os
import socket
import sys
import traceback
from typing import Any, Literal, NoReturn, Optional
import pymysql
import celery

Mode = Literal["solo", "worker", "main", "flower"]


class _TaskWrapper(celery.Task):
    max_retries = 3  # Maximum number of retries for MySQL connection errors
    default_retry_delay = 5  # 5 seconds between retries
    autoretry_for = (pymysql.OperationalError,)  # Automatically retry for MySQL connection errors
    retry_backoff = True  # Enable exponential backoff
    retry_backoff_max = 600  # Maximum retry delay of 10 minutes
    retry_jitter = True  # Add random jitter to retry delays

    def __init__(self) -> None:
        super().__init__()
        self.worker = None

    def maybe_init(self, worker_cls) -> None:
        if self.worker is None:
            self.worker = worker_cls()

    def call(self, method: str, *args, **kwargs) -> Any:
        return getattr(self.worker, method)(*args, **kwargs)

def set_mpi_env() -> None:
    global_rank = int(os.getenv("OMPI_COMM_WORLD_RANK", 0))
    local_rank = int(os.getenv("OMPI_COMM_WORLD_LOCAL_RANK", 0))
    world_size = int(os.getenv("OMPI_COMM_WORLD_SIZE", 1))

    os.environ["RANK"] = str(global_rank)
    os.environ["LOCAL_RANK"] = str(local_rank)
    os.environ["WORLD_SIZE"] = str(world_size)

def _run_worker(celery_app: celery.Celery, loglevel: str = "INFO") -> NoReturn:
    # Generating a fancy worker name with the hostname and GPU number
    hostname = socket.gethostname()
    
    set_mpi_env()
    
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    if "CUDA_VISIBLE_DEVICES" not in os.environ:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(local_rank)

    gpu = rank
    if gpu:
        hostname = f"{hostname}/{gpu}"

    celery_app.worker_main(
        [
            "worker",
            "--pool=solo",
            f"--loglevel={loglevel}",
            f"--hostname='{hostname}'",
        ]
    )
    sys.exit(0)


def _run_flower(celery_app: celery.Celery) -> NoReturn:
    try:
        import flower.app  # type: ignore
        from flower.urls import settings  # type: ignore
        from tornado.options import options
    except ImportError as e:
        print(f"Error: flower package not found. Install with 'pip install flower'. {e}")
        sys.exit(1)

    flower_app = flower.app.Flower(capp=celery_app, options=options, **settings)
    flower_app.start()
    sys.exit(0)


def setup_celery(
    name: str,
    mode: Mode,
    worker_cls: Any,
    celery_broker: Optional[str] = None,
    celery_backend: Optional[str] = None,
) -> celery.Celery:
    """
    Set up a Celery application with specific configurations based on the mode.

    The "mode" parameter is a critical argument that determines the operation of this process.

    * When "mode" is set to "solo", everything is run within the current single process.
      It executes both the workloads of the worker process and the main process by itself.
      While it cannot be parallelized, it is easy to run and convenient for development.
      In this case, launching Celery's broker/backend is not necessary.

    * When "mode" is set to "main", it is configured to be able to remotely call workers.
      In the main mode, worker_cls is not instantiated.

    * When "mode" is set to "worker", this process will subsequently act as a worker,
      and the function does not return control. The worker instantiates worker_cls,
      listens to the queue, executes tasks specified by the main process, and returns the results.

    * When "mode" is set to "flower", it launches Flower, which is Celery's web monitoring tool.
      The process does not return control from this function.

    Parameters
    ----------
    name : str
        The name of the Celery application.
    mode : Literal["solo", "worker", "main", "flower"]
        The mode in which this process is to run: 'solo', 'worker', 'main', or 'flower'.
    worker_cls : Any
        The worker class to be remotely called.
    celery_broker: Optional[str]
        The Celery broker URL. If not specified, it will be read from the environment variable "CELERY_BROKER".
        If the environment variable is not set, the default value is "pyamqp://guest@localhost//".
    celery_backend: Optional[str]
        The Celery backend URL. If not specified, it will be read from the environment variable "CELERY_BACKEND".
        If the environment variable is not set, the default value is "rpc://".

    Returns
    -------
    celery.Celery
        The configured Celery application.
    """

    if mode == "solo":
        # In the solo mode, we do not need to need an external broker.
        broker = "memory://"
    else:
        broker = celery_broker or os.environ.get("CELERY_BROKER", "pyamqp://guest@localhost:5672//")
    backend = celery_backend or os.environ.get("CELERY_BACKEND", "redis://default:user@localhost:6379/0") # rpc:// is the default backend

    app = celery.Celery(name, backend=backend, broker=broker)
    app.conf.broker_transport_options = {
        "visibility_timeout": 36000,  # 10h
        # "heartbeat": 60,  # Send heartbeat every 60 seconds to keep connection alive
    }
    app.conf.update(
        task_serializer="pickle",
        result_serializer="pickle",
        accept_content=["pickle", "json"],
        worker_prefetch_multiplier=1,
        worker_concurrency=1,
        ack_late=True,
        reject_on_worker_lost=True,
        task_default_retry_delay=0,
        task_max_retries=0,
        broker_heartbeat=None,  # Disable Celery's heartbeat, let RabbitMQ handle it
        broker_connection_retry_on_startup=True,
        broker_connection_retry=True,
        broker_connection_max_retries=10,
        task_time_limit=3600,    # 1 hour
        task_soft_time_limit=3300,  # 55 minutes
        # Add these important settings:
        result_backend_transport_options={
            'visibility_timeout': 36000,
            'socket_keepalive': True,
            'socket_keepalive_options': {
                'TCP_KEEPIDLE': 60,
                'TCP_KEEPINTVL': 10,
                'TCP_KEEPCNT': 6
            }
        },
        broker_pool_limit=None,  # Disable connection pooling issues
        worker_cancel_long_running_tasks_on_connection_loss=True,  # Address the warning you saw
    )

    def call(self: _TaskWrapper, method: str, *args, **kwargs) -> Any:
        # print(method, args, kwargs)
        try:
            self.maybe_init(worker_cls)
            return self.call(method, *args, **kwargs)
        except pymysql.OperationalError as mysql_exc:
            # For MySQL connection errors, retry with exponential backoff
            traceback.print_exc()
            sys.stderr.flush()
            sys.stdout.flush()
            raise self.retry(exc=mysql_exc)
        except Exception:  # Catch all other exceptions without binding to unused variable
            # For all other exceptions, maintain the current behavior of immediate exit
            
            # We catch all exceptions, print them out to the stderr, and exit the process.
            #
            # Otherwise, celery will catch the exception and the worker process continues to run.
            # This is not desirable, as generally exceptions are due to
            # (1) GPU-related device errors, or (2) code bugs,
            # and we want to stop the process in either case.
            #
            # Please note that, when this worker process is exited, the task is re-enqueued
            # and will be executed by another worker process.
            traceback.print_exc()
            sys.stderr.flush()
            sys.stdout.flush()

            # We use os._exit because sys.exit is hooked by Celery and does not work as expected.
            os._exit(1)

    # Register the "call" method to the task
    app.task(base=_TaskWrapper, bind=True, name="call")(call)

    if mode == "solo":
        # By setting task_always_eager to True, we can run tasks in this process.
        app.conf.task_always_eager = True
        app.conf.task_eager_propagates = True # for vscode debugger?
    elif mode == "worker":
        _run_worker(celery_app=app)
    elif mode == "flower":
        _run_flower(celery_app=app)

    # Removing the previously enqueued tasks
    app.control.purge()

    return app