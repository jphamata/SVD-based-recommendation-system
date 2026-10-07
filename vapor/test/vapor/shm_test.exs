defmodule Vapor.Runtime.ShmTest do
  use ExUnit.Case, async: false
  alias Vapor.Runtime.Shm

  @moduletag :native

  test "a file just put survives a prune by someone else (the window before a worker maps it)" do
    data = :crypto.strong_rand_bytes(4096)
    assert {:ok, path} = Shm.put(data)
    Shm.prune()
    assert File.exists?(path)
    # without the grace period it is unmapped and goes
    Shm.prune(grace: 0)
    refute File.exists?(path)
    # and a writer simply writes it again
    assert {:ok, ^path} = Shm.put(data)
    assert File.read!(path) == data
    Shm.prune(grace: 0)
  end

  test "putting an existing file starts its grace period over" do
    data = :crypto.strong_rand_bytes(1024)
    {:ok, path} = Shm.put(data)
    File.touch!(path, System.os_time(:second) - 3_600)
    {:ok, ^path} = Shm.put(data)
    Shm.prune()
    assert File.exists?(path)
    Shm.prune(grace: 0)
  end
end
