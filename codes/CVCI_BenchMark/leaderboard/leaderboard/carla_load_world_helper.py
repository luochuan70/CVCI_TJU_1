#!/usr/bin/env python3
"""Isolate CARLA's crash-prone map-load RPC from the evaluator process."""

import argparse

import carla


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", required=True)
    parser.add_argument("--port", required=True, type=int)
    parser.add_argument("--timeout", required=True, type=float)
    parser.add_argument("--town", required=True)
    parser.add_argument("--frame-rate", required=True, type=float)
    args = parser.parse_args()

    client = carla.Client(args.host, args.port)
    client.set_timeout(args.timeout)
    world = client.load_world(args.town, reset_settings=False)

    settings = world.get_settings()
    settings.synchronous_mode = True
    settings.fixed_delta_seconds = 1.0 / args.frame_rate
    settings.deterministic_ragdolls = True
    settings.spectator_as_ego = False
    settings.tile_stream_distance = 650
    settings.actor_active_distance = 650
    world.apply_settings(settings)
    world.reset_all_traffic_lights()
    world.tick()

    map_name = world.get_map().name.split("/")[-1]
    if map_name != args.town:
        raise RuntimeError(
            "CARLA loaded map {!r}, expected {!r}".format(map_name, args.town)
        )


if __name__ == "__main__":
    main()
