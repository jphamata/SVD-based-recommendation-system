defmodule Vapor.Finance.Fix do
  @moduledoc """
  FIX 4.4 tag=value messages (docs/FINANCE.md §10): parse with
  BodyLength (9) and CheckSum (10) verified, encode with both computed,
  and a gateway that turns NewOrderSingle (D), OrderCancelRequest (F)
  and OrderCancelReplaceRequest (G) into book events and the book's
  reports into ExecutionReports (8). The byte rules are checked against
  `simplefix` in `finance_test.exs`.

  The separator is SOH (0x01); `|` is accepted on input for readability.
  """
  @soh <<1>>

  @doc "Parse one message. `{:ok, [{tag, value}…]}` or `{:error, why}` (BodyLength and CheckSum verified)."
  def parse(msg) do
    msg = String.replace(msg, "|", @soh)
    msg = if String.ends_with?(msg, @soh), do: msg, else: msg <> @soh
    fields = msg |> String.split(@soh, trim: true) |> Enum.map(fn f -> case String.split(f, "=", parts: 2) do [t, v] -> {t, v}; _ -> {:bad, f} end end)
    cond do
      Enum.any?(fields, &match?({:bad, _}, &1)) -> {:error, "a field is not tag=value"}
      Enum.any?(fields, fn {t, _} -> not (t =~ ~r/^\d+$/) end) -> {:error, "a tag is not a number"}
      length(fields) < 4 -> {:error, "too short for a FIX message"}
      elem(Enum.at(fields, 0), 0) != "8" or elem(Enum.at(fields, 1), 0) != "9" or elem(List.last(fields), 0) != "10" -> {:error, "a message starts with 8= and 9= and ends with 10="}
      true ->
        fields = Enum.map(fields, fn {t, v} -> {String.to_integer(t), v} end)
        # BodyLength: bytes after the 9= field's SOH up to (and including) the SOH before 10=
        [_, start_and_rest] = String.split(msg, @soh <> "9=", parts: 2)
        [_len, after9] = String.split(start_and_rest, @soh, parts: 2)
        body = binary_part(after9, 0, byte_size(after9) - byte_size(trailer_of(after9)))
        declared = fields |> List.keyfind(9, 0) |> elem(1)
        sum_part = binary_part(msg, 0, byte_size(msg) - byte_size(trailer_of(msg)))
        cks = checksum(sum_part)
        declared_ck = fields |> List.keyfind(10, 0) |> elem(1)
        cond do
          Integer.to_string(byte_size(body)) != declared -> {:error, "BodyLength #{declared} but the body has #{byte_size(body)} bytes"}
          cks != declared_ck -> {:error, "CheckSum #{declared_ck} but the bytes sum to #{cks}"}
          true -> {:ok, fields}
        end
    end
  end

  defp trailer_of(s) do
    case :binary.matches(s, "10=") |> List.last() do
      {pos, _} -> binary_part(s, pos, byte_size(s) - pos)
      nil -> ""
    end
  end

  @doc "The three-digit CheckSum of the bytes (sum mod 256)."
  def checksum(bytes), do: bytes |> :binary.bin_to_list() |> Enum.sum() |> rem(256) |> Integer.to_string() |> String.pad_leading(3, "0")

  @doc "Encode: `[{tag, value}]` body fields after 35=type; header 8/9 and trailer 10 computed."
  def encode(type, fields, opts \\ []) do
    head = [{35, type}, {49, Keyword.get(opts, :sender, "CLIENT")}, {56, Keyword.get(opts, :target, "VAPOR")}, {34, Keyword.get(opts, :seq, 1)}] ++
      if(Keyword.get(opts, :time), do: [{52, Keyword.get(opts, :time)}], else: [])
    body = Enum.map_join(head ++ fields, "", fn {t, v} -> "#{t}=#{v}" <> @soh end)
    pre = "8=FIX.4.4" <> @soh <> "9=#{byte_size(body)}" <> @soh <> body
    pre <> "10=" <> checksum(pre) <> @soh
  end

  def readable(msg), do: String.replace(msg, @soh, "|")

  # ------------------------------------------------------------- gateway

  @tif %{"0" => :gtc, "1" => :gtc, "3" => :ioc, "4" => :fok}

  @doc """
  A FIX message → a book event: `{:ok, event}` or `{:error, why}`.
  Prices are converted to integer ticks with `tick` (default 0.01).
  """
  def to_event(fields, opts \\ []) do
    tick = Keyword.get(opts, :tick, 0.01)
    g = fn t -> case List.keyfind(fields, t, 0) do {_, v} -> v; nil -> nil end end
    ticks = fn nil -> nil; p -> round(String.to_float(if String.contains?(p, "."), do: p, else: p <> ".0") / tick) end
    owner = g.(49) || g.(1) || "0"
    case g.(35) do
      "D" ->
        with {qty, ""} <- Integer.parse(g.(38) || ""), side when side in ["1", "2"] <- g.(54), ot when ot in ["1", "2"] <- g.(40) do
          {:ok, %{type: :new, id: g.(11), owner: owner, side: if(side == "1", do: :buy, else: :sell), qty: qty,
                  price: if(ot == "2", do: ticks.(g.(44)), else: nil), tif: Map.get(@tif, g.(59) || "0", :gtc),
                  post_only: String.contains?(g.(18) || "", "6")}}
        else
          _ -> {:error, "NewOrderSingle needs 11, 54 (1|2), 38, 40 (1|2) and 44 for limits"}
        end
      "F" -> {:ok, %{type: :cancel, id: g.(41)}}
      "G" ->
        case Integer.parse(g.(38) || "") do
          {qty, ""} -> {:ok, %{type: :modify, id: g.(41), price: ticks.(g.(44)), qty: qty}}
          _ -> {:error, "OrderCancelReplaceRequest needs 41 and 38"}
        end
      t -> {:error, "message type #{inspect(t)}: the gateway takes D, F and G"}
    end
  end

  @doc """
  Book reports → ExecutionReports (35=8). `ExecType`: 0 new, F trade,
  4 cancelled, 5 replaced, 8 rejected; `OrdStatus` from what is left.
  """
  def exec_reports(reports, _orders, opts \\ []) do
    tick = Keyword.get(opts, :tick, 0.01)
    start = Keyword.get(opts, :seq, 1)
    bodies =
      Enum.flat_map(reports, fn rep ->
        case rep do
          {:accepted, id} -> [{id, [{150, "0"}, {39, "0"}]}]
          {:fill, f} ->
            px = :erlang.float_to_binary(f.price * tick, decimals: 2)
            [{f.taker, [{150, "F"}, {39, "1"}, {31, px}, {32, f.qty}]}, {f.maker, [{150, "F"}, {39, "1"}, {31, px}, {32, f.qty}]}]
          {:cancelled, id, q, why} -> [{id, [{150, if(why == :replaced, do: "5", else: "4")}, {39, "4"}, {151, 0}, {58, "#{why}; #{q} left"}]}]
          {:modified, id, q} -> [{id, [{150, "5"}, {39, "0"}, {151, q}]}]
          {:rejected, id, why} -> [{id || "?", [{150, "8"}, {39, "8"}, {58, to_string(why)}]}]
          {:risk_rejected, id, why} -> [{id || "?", [{150, "8"}, {39, "8"}, {58, "risk: #{why}"}]}]
          _ -> []
        end
      end)
    # every message its own MsgSeqNum (34) and ExecID (17)
    bodies |> Enum.with_index(start) |> Enum.map(fn {{id, fields}, n} -> encode("8", [{37, id}, {11, id}, {17, "E#{n}"}] ++ fields, Keyword.put(opts, :seq, n)) end)
  end
end
