defmodule Vapor.Runtime.Shm do
  @moduledoc """
  Content-addressed shared memory for weights (`/dev/shm/vapor-<sha256>`).

  Written once (atomically: temp file + rename), then mapped read-only by
  every worker and by the fabric daemon — the zero-copy path for tensors
  that dominate memory traffic. Content addressing makes the store safe to
  share between nodes and restarts: a name *is* its bytes.

  `prune/1` removes the files no process maps (a file that is mapped stays:
  Linux keeps an unlinked mapping alive, but a worker restarting later
  would need the name) **and** that nobody has `put` recently — a grace
  period, because between `put/1` returning a path and a worker mapping it
  there is a window in which the file is unmapped but about to be used.
  (0.15: two suites sharing `/dev/shm` hit that window — one's `prune` removed
  a file the other had just written, and its worker failed to open it.)
  `put/1` refreshes the time of a file that already exists.
  """
  @dir "/dev/shm"
  @grace 600

  @doc """
  Remove every `vapor-*` file no process currently maps and nobody has
  `put` in the last `grace` seconds (default #{@grace}): `%{removed, kept, bytes}`.
  """
  def prune(opts \\ []) do
    grace = Keyword.get(opts, :grace, @grace)
    now = System.os_time(:second)
    mapped =
      case File.ls("/proc") do
        {:ok, pids} ->
          for pid <- pids, String.match?(pid, ~r/^\d+$/), {:ok, maps} <- [File.read("/proc/#{pid}/maps")],
              line <- String.split(maps, "\n"), String.contains?(line, @dir <> "/vapor-"), into: MapSet.new() do
            line |> String.split() |> List.last()
          end

        _ -> MapSet.new()
      end

    files = case File.ls(@dir) do
      {:ok, fs} -> for f <- fs, String.starts_with?(f, "vapor-"), do: Path.join(@dir, f)
      _ -> []
    end

    recent? = fn f ->
      case File.stat(f, time: :posix) do
        {:ok, %{mtime: t}} -> now - t < grace
        _ -> false
      end
    end

    {gone, kept} = Enum.split_with(files, &(not MapSet.member?(mapped, &1) and not recent?.(&1)))
    bytes = gone |> Enum.map(&(case File.stat(&1) do {:ok, st} -> st.size; _ -> 0 end)) |> Enum.sum()
    Enum.each(gone, &File.rm/1)
    %{removed: length(gone), kept: length(kept), bytes: bytes}
  end

  @spec put(binary) :: {:ok, String.t()} | :unavailable
  def put(data) do
    if File.dir?(@dir) do
      path = Path.join(@dir, "vapor-" <> Base.encode16(:crypto.hash(:sha256, data), case: :lower))

      if File.exists?(path) do
        # in use again: the grace period of prune/1 starts over
        _ = File.touch(path)
        {:ok, path}
      else
        tmp = path <> ".#{System.unique_integer([:positive])}.tmp"

        with :ok <- File.write(tmp, data), :ok <- File.rename(tmp, path) do
          {:ok, path}
        else
          _ -> :unavailable
        end
      end
    else
      :unavailable
    end
  end
end
