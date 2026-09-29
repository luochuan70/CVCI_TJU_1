# Scene-selected runtime code

`radar_model_switch_agent.py` selects the runtime rule stack after the first
valid `RADAR_FRONT` packet and the official route-reference match:

- `extreme_144`: use the copied 0725 `cvci_auxiliary_system.py` and historical
  dispatch module under `extreme_144/`.
- `ordinary_220`: use the 0810-compatible 0629 auxiliary and historical
  dispatch modules; the 0824 ordinary-scene additions are not loaded.

The shared model, planner, PID, perception, and scenario-recognition modules
are unchanged because their 0725 and 0819 versions are byte-identical.  The
copied files are retained in this package so the 144 path does not depend on
the external 0725 checkout at runtime.

For full-set evaluation, `run_all_routes.sh` selects the official
`../route/CVCI_BenchMark.xml` or `../route/bench2drive220.xml` through
`CVCI_ROUTE_SET`; explicit `CVCI_ROUTES_FILE` still allows a single-route or
mixed XML.
