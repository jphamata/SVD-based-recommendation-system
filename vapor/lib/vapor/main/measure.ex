defmodule Vapor.Main.Measure do
  @moduledoc """
  The one place the product runs an **operator-named** command: the
  Athanor's external measure (`bin/vapor athanor run --measure CMD`). It is
  the same category as the MCP client's tool server — a program the person
  at the terminal names, never code the compiler generated — and it is
  confined the same way: a port to `/bin/sh` with the candidate in the
  environment, a **deadline** (a hung measure is a failed evaluation, not a
  frozen furnace), a cap on what it may print, and the port closed (its OS
  process killed) on every path out.

  (0.14 shelled out here directly, without a deadline, against the audit's
  own rule; the audit test caught it in 0.15.)
  """

  @max_output 1_048_576

  @doc """
  Run `cmd` with extra environment `env` (`[{name, value}]`): `{:ok, stdout}`
  when it exits 0 within `timeout` ms (default 60 000), otherwise
  `{:error, why}`.
  """
  def run(cmd, env, opts \\ []) when is_binary(cmd) do
    timeout = Keyword.get(opts, :timeout, 60_000)

    case System.find_executable("sh") do
      nil ->
        {:error, "no sh on this machine"}

      sh ->
        # the deadline is enforced by the shell itself (a watchdog kills the
        # command); the receive below is the second barrier
        secs = div(timeout + 999, 1000)
        # The command runs in its own session (setsid, where present) so the
        # watchdog ends its whole process group, grandchildren included; the
        # watchdog writes nowhere, so it never holds the output pipe open, and
        # ending it ends its sleep.
        start = ~s|if command -v setsid >/dev/null 2>&1; then setsid sh -c "$VAPOR_MEASURE_CMD" & c=$!; g=-$c; else ( eval "$VAPOR_MEASURE_CMD" ) & c=$!; g=$c; fi|
        watchdog = ~s|( sleep #{secs} & s=$!; trap "kill $s 2>/dev/null; exit 0" TERM; wait $s; kill -9 $g 2>/dev/null ) >/dev/null 2>&1|
        script = ~s|#{start}; #{watchdog} & w=$!; wait $c 2>/dev/null; s=$?; kill $w 2>/dev/null; exit $s|
        env = [{"VAPOR_MEASURE_CMD", cmd} | env]

        port =
          Port.open({:spawn_executable, sh}, [:binary, :exit_status, :use_stdio, :hide, args: ["-c", script],
                                              env: Enum.map(env, fn {k, v} -> {String.to_charlist(k), String.to_charlist(v)} end)])

        deadline = System.monotonic_time(:millisecond) + timeout

        try do
          collect(port, [], 0, deadline)
        after
          close(port)
        end
    end
  end

  defp collect(port, acc, size, deadline) do
    left = max(deadline - System.monotonic_time(:millisecond), 0)

    receive do
      {^port, {:data, d}} ->
        size = size + byte_size(d)
        if size > @max_output, do: {:error, "the command printed more than #{@max_output} bytes"}, else: collect(port, [acc, d], size, deadline)

      {^port, {:exit_status, 0}} ->
        {:ok, IO.iodata_to_binary(acc)}

      {^port, {:exit_status, code}} ->
        {:error, "the command exited with #{code}"}
    after
      (left + 2_000) -> {:error, "the command did not finish within the deadline"}
    end
  end

  defp close(port) do
    if Port.info(port) != nil, do: Port.close(port)
    flush(port)
  catch
    :error, _ -> :ok
  end

  defp flush(port) do
    receive do
      {^port, _} -> flush(port)
    after
      0 -> :ok
    end
  end
end
