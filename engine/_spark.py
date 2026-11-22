"""One Databricks Connect (serverless) session for the engine. Locally: DATABRICKS_CONFIG_PROFILE=datathon."""
import os

import certifi

os.environ.setdefault("GRPC_DEFAULT_SSL_ROOTS_FILE_PATH", certifi.where())
from databricks.connect import DatabricksSession  # noqa: E402

spark = DatabricksSession.builder.serverless(True).getOrCreate()
