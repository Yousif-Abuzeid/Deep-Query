from prometheus_client import Counter, Histogram,generate_latest, CONTENT_TYPE_LATEST
from fastapi import Response,FastAPI, Request
from starlette.middleware.base import BaseHTTPMiddleware
import time


# Define Prometheus metrics
REQUEST_COUNT = Counter('http_requests_total', 'Total HTTP Requests', ['method', 'endpoint', 'status'])
REQUEST_LATENCY = Histogram('http_request_duration_seconds', 'HTTP Request Latency', ['method', 'endpoint'])

class PrometheusMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        start_time = time.time()
        response = await call_next(request)
        duration = time.time() - start_time
        

        REQUEST_COUNT.labels(method=request.method, endpoint=request.url.path, status=response.status_code).inc()
        REQUEST_LATENCY.labels(method=request.method, endpoint=request.url.path).observe(duration)

        return response
    
def setup_metrics(app: FastAPI):
    """
    Setup Prometheus metrics endpoint and middleware for FastAPI app.
    """
    app.add_middleware(PrometheusMiddleware)

    @app.get("/fnjkasdjkfndsajf", include_in_schema=False)
    def metrics():
        return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)