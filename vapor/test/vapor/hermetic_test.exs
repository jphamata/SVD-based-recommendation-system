defmodule Vapor.HermeticTest do
  use ExUnit.Case, async: true
  alias Vapor.Hermetic

  # 512 MB built as 1 MB binaries: all of it lives off the process heap
  defp binary_bomb, do: for(_ <- 1..512, do: :binary.copy(<<0>>, 1_048_576)) |> length()

  test "a job returns its value" do
    assert {:ok, 42} = Hermetic.seal(fn -> 6 * 7 end)
  end

  test "off-heap binaries count against the cap (the 0.16 sandboxes let them through)" do
    assert {:error, :memory} = Hermetic.seal(&binary_bomb/0, heap_mb: 64)

    # the control: the same job under the old heap-only cap survives with 8× the cap allocated
    words = div(64 * 1_048_576, :erlang.system_info(:wordsize))
    {pid, mon} = spawn_monitor(fn ->
      Process.flag(:max_heap_size, %{size: words, kill: true, error_logger: false})
      exit({:survived, binary_bomb()})
    end)

    assert_receive {:DOWN, ^mon, :process, ^pid, {:survived, 512}}, 30_000
  end

  test "a heap bomb is stopped too" do
    assert {:error, :memory} = Hermetic.seal(fn -> Enum.to_list(1..50_000_000) |> length() end, heap_mb: 16)
  end

  test "a deadline stops a job, and its late reply never reaches the caller" do
    assert {:error, :timeout} = Hermetic.seal(fn -> Process.sleep(2_000); :late end, timeout: 50)
    refute_receive {_, :late}, 100
  end

  test "a crash comes back as a value; the caller survives" do
    assert {:error, {:crash, {%ArgumentError{}, _}}} = Hermetic.seal(fn -> raise ArgumentError, "boom" end)
    assert Process.alive?(self())
  end

  test "failures read as one line" do
    assert Hermetic.describe(:memory, heap_mb: 64) == "used more than 64 MB and was stopped"
    assert Hermetic.describe(:timeout, timeout: 30_000) == "took longer than 30 s and was stopped"
    assert {:error, f} = Hermetic.seal(fn -> raise ArgumentError, "boom" end)
    assert Hermetic.describe(f) == "stopped: boom"
  end

  test "cap_self caps a long-lived process the same way" do
    {pid, mon} = spawn_monitor(fn -> Hermetic.cap_self(64); binary_bomb() end)
    assert_receive {:DOWN, ^mon, :process, ^pid, :killed}, 30_000
  end

  test "document ingestion is sealed: a file that costs more than the seal is refused, not an outage" do
    text = String.duplicate("the quick brown fox jumps over the lazy dog\n", 200_000)
    assert {:ok, %{passages: [_ | _]}} = Vapor.Docs.ingest({"fox.txt", text})
    assert {:error, %Vapor.Rejection{bound: bound}} = Vapor.Docs.ingest({"fox.txt", text}, heap_mb: 4)
    assert bound =~ "used more than 4 MB"
  end
end
