defmodule Vapor.CLI do
  @moduledoc false
  # shared option handling of the mix tasks

  @doc """
  A command-line argument as the user typed it. Without a UTF-8 locale
  (`LANG` unset, common in containers) the BEAM reads argv as Latin-1, so
  "família" arrives as "famÃ\u00adlia". When every code point fits a byte and
  those bytes are valid UTF-8 with at least one multibyte character, the
  argument is decoded again; anything else is left as it is.
  """
  def utf8_arg(s) when is_binary(s) do
    cps = String.to_charlist(s)

    if Enum.all?(cps, &(&1 <= 255)) and Enum.any?(cps, &(&1 >= 0xC2)) do
      bytes = :binary.list_to_bin(cps)
      if String.valid?(bytes) and bytes != s, do: bytes, else: s
    else
      s
    end
  end

  def utf8_arg(other), do: other

  def engine(dir, tk, o) do
    opts =
      [model: dir, tokenizer: tk, threads: o[:threads] || System.schedulers_online(),
       quantize: if(o[:quantize] == "sb4", do: :sb4), storage: if(o[:storage] == "bf16", do: :bf16, else: :f32),
       model_name: Path.basename(Path.expand(dir))] ++
        (if o[:gpu], do: [isa: :spirv], else: []) ++
        Enum.flat_map([:max_seq, :sequences, :page, :step_tokens, :replicas], fn k -> if o[k], do: [{k, o[k]}], else: [] end)

    case Vapor.Engine.start_link(opts) do
      {:ok, e} -> e
      {:error, r} -> Mix.raise("could not start the engine: #{inspect(r)}")
    end
  end
end
