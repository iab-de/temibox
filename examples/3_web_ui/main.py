import os
import sys
sys.path.append(f"{os.getcwd()}/examples/3_web_ui")

from web.app import app, logger
from gevent.pywsgi import WSGIServer

# Server configuration
host = "0.0.0.0"
port = 8882

# Start server
logger.info(f"Serving at {host}:{port}")

http_server = WSGIServer((host, port), app)
http_server.serve_forever()