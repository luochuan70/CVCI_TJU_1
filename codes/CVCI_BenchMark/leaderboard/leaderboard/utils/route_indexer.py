from collections import OrderedDict
from dictor import dictor

import copy

from leaderboard.utils.route_parser import RouteParser
from leaderboard.utils.checkpoint_tools import fetch_dict


class RouteIndexer():
    def __init__(self, routes_file, repetitions, routes_subset):
        self._configs_dict = OrderedDict()
        self._configs_list = []
        self.index = 0

        route_configurations = RouteParser.parse_routes_file(routes_file, routes_subset)
        self.total = len(route_configurations) * repetitions

        for i, config in enumerate(route_configurations):
            for repetition in range(repetitions):
                config.index = i * repetitions + repetition
                config.repetition_index = repetition
                self._configs_dict['{}.{}'.format(config.name, repetition)] = copy.copy(config)

        self._configs_list = list(self._configs_dict.values())


    def peek(self):
        return self.index < self.total

    def get_next_config(self):
        if self.index >= self.total:
            return None

        config = self._configs_list[self.index]
        self.index += 1

        return config

    def validate_and_resume(self, endpoint):
        """
        Validates the endpoint by comparing several of its values with the current running routes.
        If all checks pass, the simulation starts from the last route.
        Otherwise, the resume is canceled, and the leaderboard goes back to normal behavior
        """
        data = fetch_dict(endpoint)
        if not data:
            print('Problem reading checkpoint. Found no data')
            return False

        entry_status = dictor(data, 'entry_status')
        if not entry_status:
            print("Problem reading checkpoint. Given checkpoint is malformed")
            return False
        if entry_status == "Invalid":
            print("Problem reading checkpoint. The 'entry_status' is 'Invalid'")
            return False

        checkpoint_dict = dictor(data, '_checkpoint')
        if not checkpoint_dict or 'progress' not in checkpoint_dict:
            print("Problem reading checkpoint. Given endpoint is malformed")
            return False

        progress = checkpoint_dict['progress']
        if progress[1] != self.total:
            print("Problem reading checkpoint. Endpoint's amount of routes does not match the given one")
            return False

        route_data = dictor(checkpoint_dict, 'records') or []

        # Resume from the already persisted records, not just from the saved
        # progress cursor. CARLA/evaluator crashes can leave progress behind
        # records; in that case using progress repeats routes that already have
        # terminal statistics. For this benchmark run, any route with a matching
        # checkpoint record is considered tested and must be skipped on restart.
        record_resume_index = 0
        for check_index, record in enumerate(route_data[:self.total]):
            route_id = self._configs_list[check_index].name
            route_id += "_rep" + str(self._configs_list[check_index].repetition_index)
            checkpoint_route_id = record.get('route_id')

            if route_id != checkpoint_route_id:
                print("Problem reading checkpoint. Checkpoint routes don't match the current ones")
                return False

            record_resume_index = check_index + 1

        if record_resume_index:
            resume_index = record_resume_index
        else:
            resume_index = min(progress[0], self.total)

        self.index = max(0, min(resume_index, self.total))
        return True

