defmodule Vapor.Runtime.Shm do
  @moduledoc """
  Content-addressed shared memory for weights (`/dev/shm/vapor-<sha256>`).

  Written once (atomically: temp file + rename), then mapped read-only by
  every worker and by the fabric daemon — the zero-copy path for tensors
  that dominate memory traffic. Content addressing makes the store safe to
  share between nodes and restarts: a name *is* its bytes.

  `prune/0` removes the files no process maps (a file that is mapped stays:
  Linux keeps an unlinked mapping alive, but a worker restarting later
  would need the name). A writer whose file was pruned simply writes it
  again: `put/1` checks for the name each time.
  """
  @dir "/dev/shm"

  @doc "Remove every `vapor-*` file no process currently maps: `%{removed, kept, bytes}`."
  def prune do
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

    {gone, kept} = Enum.split_with(files, &(not MapSet.member?(mapped, &1)))
    bytes = gone |> Enum.map(&(File.stat!(&1).size)) |> Enum.sum()
    Enum.each(gone, &File.rm/1)
    %{removed: length(gone), kept: length(kept), bytes: bytes}
  end

  @spec put(binary) :: {:ok, String.t()} | :unavailable
  def put(data) do
    if File.dir?(@dir) do
      path = Path.join(@dir, "vapor-" <> Base.encode16(:crypto.hash(:sha256, data), case: :lower))

      if File.exists?(path) do
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
