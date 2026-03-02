import argparse
import time
from datetime import datetime, timedelta

from sandbagging_workshop.inference import Client, cedar, maple, trusted_reference, yew

ENDPOINTS = {
    "cedar": cedar,
    "yew": yew,
    "maple": maple,
    "trusted_reference": trusted_reference,
}


def keep_alive(endpoint_name: str, interval_minutes: int = 30) -> None:
    endpoint = ENDPOINTS[endpoint_name]
    client = Client(endpoint=endpoint)

    print(f"Keeping {endpoint_name} alive (pinging every {interval_minutes} minutes)")
    print("Press Ctrl+C to stop\n")

    try:
        while True:
            now = datetime.now()
            print(f"[{now:%H:%M:%S}] Pinging {endpoint_name}...")
            client.wake_up()
            next_ping = now + timedelta(minutes=interval_minutes)
            print(f"[{now:%H:%M:%S}] Done. Next ping at {next_ping:%H:%M:%S}\n")
            time.sleep(interval_minutes * 60)
    except KeyboardInterrupt:
        print(f"\nStopped keep-alive for {endpoint_name}")


def main():
    parser = argparse.ArgumentParser(description="Keep a HuggingFace inference endpoint alive")
    parser.add_argument("endpoint", choices=ENDPOINTS.keys(), help="Endpoint name to keep alive")
    parser.add_argument("--interval", type=int, default=15, help="Minutes between pings (default: 15)")
    args = parser.parse_args()
    keep_alive(args.endpoint, args.interval)


if __name__ == "__main__":
    main()
