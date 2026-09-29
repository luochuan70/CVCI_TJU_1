# Method summary

The base policy is DriveTransformer. A live front radar stream is processed over a temporal window into distance, azimuth, radial-velocity, occupancy, density, approach and TTC-related features. The ordinary classifier emits a class, confidence and evidence; temporal confirmation and route-local state prevent rapid category switching. Evidence that is insufficient for a class lock uses a conservative fallback.

`OrdinaryRuleRouter` owns one fresh classifier and one fresh instance of every category rule for each route. It routes the online classification to the corresponding `class_01` through `class_44` module. Category modules share common bounded safety and local-path primitives but keep independent configurations and state machines. The online local planner can stage, wait for a traffic gap, execute a short obstacle bypass and rejoin the base route. No category rule branches on a route ID or route filename.

Extreme routes use the separately packaged, frozen extreme-144 rule stack and checkpoint. They never instantiate the ordinary classifier or router.

