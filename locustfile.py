from locust import HttpUser, task, between

class StreamlitUser(HttpUser):
    wait_time = between(1, 3)

    @task
    def load_main_page(self):
        # Streamlit standard entry points
        self.client.get("/")
        self.client.get("/_stcore/health")
        
    @task(3)
    def simulate_prediction(self):
        # Simulate websocket / backend inference latency request 
        # For a standard STREAMLIT setup, we ping the health/message endpoints
        self.client.get("/_stcore/health")
