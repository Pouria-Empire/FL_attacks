import argparse
import yaml
import flwr as fl

def load_config():
    """Loads the main config.yml file."""
    with open("config.yml", "r") as f:
        return yaml.safe_load(f)

def main():
    """Parses arguments and starts the appropriate client."""
    parser = argparse.ArgumentParser(description="Flower Client Launcher")
    parser.add_argument("--cid", type=str, required=True, help="Client ID (e.g., client1)")
    parser.add_argument(
        "--type",
        type=str,
        required=True,
        choices=['image', 'sensor'],
        help="Type of client to run: 'image' or 'sensor'"
    )
    args = parser.parse_args()

    config = load_config()
    client_to_run = None

    if args.type == 'image':
        from clients.image_client import ImageFlowerClient
        print(f"Starting IMAGE client {args.cid}...")
        client_to_run = ImageFlowerClient(args.cid, config)
    elif args.type == 'sensor':
        from clients.sensor_client import SensorFlowerClient
        print(f"Starting SENSOR client {args.cid}...")
        client_to_run = SensorFlowerClient(args.cid, config)

    # Start the selected Flower client
    fl.client.start_numpy_client(
        server_address="127.0.0.1:8080",
        client=client_to_run,
        grpc_max_message_length=1024*1024*1024
    )

if __name__ == "__main__":
    main()