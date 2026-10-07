defmodule Vapor.Entropy do
  @moduledoc """
  The **one entropy boundary** (ASAS §6, "determinism by construction,
  entropy at a single boundary"). Everything internal to vapor is a
  deterministic function of its inputs — seeded generators, hashes — so a
  run can be replayed and a fault reproduced. True entropy is admitted here
  and nowhere else, and only where an adversary forces unpredictability.

  The sanctioned uses of `bytes/1`, each because a party who could predict
  the value would gain something:

    * a store's secret key (`Vapor.Khazana`), which makes computed
      capabilities unforgeable;
    * a data subject's sealing key (`Vapor.Agent.Keys`), which makes erasure
      by key shredding real;
    * a run nonce (`Vapor.Agent`), so two runs of the same agent on the same
      input at the same instant still have distinct identities;
    * identifiers that act as bearer handles (an Athanor session, a
      completion id, a store's write tag), which must not be guessable;
    * the audit salt of `Vapor.Cluster`, so a node cannot predict which of
      its answers will be re-checked.

  Everything else uses `rng/1`: a functional generator (`:rand` state is
  passed and returned, never kept in the process dictionary) seeded by
  hashing its parts, so the same parts give the same stream on every
  machine. `test/vapor/audit_test.exs` enforces both rules over `lib/`.
  """

  @doc "`n` bytes from the operating system's CSPRNG — the only call that reaches it."
  def bytes(n) when is_integer(n) and n > 0, do: :crypto.strong_rand_bytes(n)

  @doc "An unguessable URL-safe token of `n` random bytes."
  def token(n \\ 12), do: Base.url_encode64(bytes(n), padding: false)

  @doc """
  A deterministic generator state from any term: the SHA-256 of its
  canonical encoding seeds `:exsss`. `rng({:shuffle, 7})` is the same
  stream everywhere.
  """
  def rng(parts) do
    <<a::64, b::64, c::64, _::binary>> = :crypto.hash(:sha256, :erlang.term_to_binary(parts, [:deterministic]))
    :rand.seed_s(:exsss, {a, b, c})
  end

  @doc "A uniform integer in `1..n` and the next state."
  def uniform(n, rng) when is_integer(n) and n > 0, do: :rand.uniform_s(n, rng)

  @doc "A uniform float in `[0, 1)` and the next state."
  def float(rng), do: :rand.uniform_s(rng)

  @doc "A standard normal and the next state."
  def normal(rng), do: :rand.normal_s(rng)

  @doc "The list shuffled (Fisher–Yates by sort keys) and the next state."
  def shuffle(list, rng) do
    {keyed, rng} = Enum.map_reduce(list, rng, fn x, r -> {u, r} = :rand.uniform_s(r); {{u, x}, r} end)
    {keyed |> Enum.sort_by(&elem(&1, 0)) |> Enum.map(&elem(&1, 1)), rng}
  end

  @doc "One element at random and the next state."
  def pick([_ | _] = list, rng) do
    {i, rng} = :rand.uniform_s(length(list), rng)
    {Enum.at(list, i - 1), rng}
  end

  @doc "A shuffle for one-off use where the state is not needed afterwards."
  def shuffled(list, parts), do: list |> shuffle(rng(parts)) |> elem(0)
end
