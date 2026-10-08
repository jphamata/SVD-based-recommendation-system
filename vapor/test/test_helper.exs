# Tiers are tested exactly when their tooling exists on this machine; the
# control plane (algebra, emitters, allocator, checker, oracle, envelope,
# certificates) is always tested. The probes are Vapor.TestTiers.excluded/0,
# shared with `mix vapor.test` (whose cache keys include them).
exclude = Vapor.TestTiers.excluded()

if exclude != [], do: IO.puts("vapor: skipping tiers without tooling: #{inspect(exclude)}")
# logs (a contained worker crash, a killed engine) are shown only for a failing test
ExUnit.start(exclude: exclude, capture_log: true)
# leave no orphaned weight files in shared memory
ExUnit.after_suite(fn _ -> Vapor.Runtime.Shm.prune() end)
