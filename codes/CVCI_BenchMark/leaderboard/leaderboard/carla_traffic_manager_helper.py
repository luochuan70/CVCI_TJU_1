#!/usr/bin/env python3
"""Own CARLA TrafficManager C++ handles outside the evaluator process."""

import argparse
import sys

import carla


def acknowledge(message="ACK"):
    print(message, flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", required=True)
    parser.add_argument("--rpc-port", required=True, type=int)
    parser.add_argument("--tm-port", required=True, type=int)
    parser.add_argument("--timeout", required=True, type=float)
    args = parser.parse_args()

    client = carla.Client(args.host, args.rpc_port)
    client.set_timeout(args.timeout)
    traffic_manager = client.get_trafficmanager(args.tm_port)
    traffic_manager.set_synchronous_mode(True)
    traffic_manager.set_hybrid_physics_mode(True)
    acknowledge("READY")

    for raw_line in sys.stdin:
        parts = raw_line.strip().split()
        if not parts:
            continue
        command = parts[0]
        if command == "seed" and len(parts) == 2:
            traffic_manager.set_random_device_seed(int(parts[1]))
            acknowledge()
        elif command == "async":
            traffic_manager.set_synchronous_mode(False)
            traffic_manager.set_hybrid_physics_mode(False)
            acknowledge()
        elif command == "shutdown":
            traffic_manager.set_synchronous_mode(False)
            traffic_manager.set_hybrid_physics_mode(False)
            traffic_manager.shut_down()
            acknowledge()
            return
        else:
            acknowledge("ERROR unsupported command")


if __name__ == "__main__":
    main()
