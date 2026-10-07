defmodule Vapor.Chat do
  @moduledoc """
  Chat messages → prompt text.

  With the model's own chat template (`load_template/1`: `chat_template.jinja`,
  the `chat_template` of `tokenizer_config.json`, or a GGUF's
  `tokenizer.chat_template`) the prompt is rendered by `Vapor.Template`
  exactly as transformers would render it — tools, tool calls and tool
  results included (`render/3`). Without one, the format is recognised from
  the vocabulary's special tokens (`render/2`):

    * `<|im_start|>` — ChatML (Qwen2): `<|im_start|>role\\ncontent<|im_end|>\\n`…,
      then `<|im_start|>assistant\\n`; the turn ends at `<|im_end|>`;
    * `<|start_header_id|>` — Llama 3: `<|begin_of_text|>` and
      `<|start_header_id|>role<|end_header_id|>\\n\\ncontent<|eot_id|>`…;
      the turn ends at `<|eot_id|>`;
    * otherwise the Llama 2 / Mistral form `[INST] … [/INST]` (a system
      message folded into the first user turn), ending at EOS.

  Returns `{prompt, add_bos, stop_token_ids}`.
  """
  alias Vapor.Tokenizer

  @end_of_turn ~w(<|im_end|> <|eot_id|> <|eom_id|> <|end|> <end_of_turn> <|endoftext|> <|end_of_text|> </s> <|return|> <|call|> <｜end▁of▁sentence｜>)

  @doc """
  The chat template of a checkpoint directory or `.gguf` file:
  `{:ok, %{source, template, bos, eos}}` (`bos`/`eos` the token strings the
  template may print) or `{:error, :none}`.
  """
  def load_template(path) do
    {src, cfg} =
      cond do
        File.regular?(path) and String.ends_with?(path, ".gguf") ->
          case Vapor.Ingest.GGUF.read(path) do
            {:ok, g} -> {g.metadata["tokenizer.chat_template"], %{}}
            _ -> {nil, %{}}
          end

        true ->
          cfg = case File.read(Path.join(path, "tokenizer_config.json")) do
            {:ok, b} -> Vapor.JSON.decode!(b)
            _ -> %{}
          end

          jinja = Path.join(path, "chat_template.jinja")
          {if(File.regular?(jinja), do: File.read!(jinja), else: pick(cfg["chat_template"])), cfg}
      end

    with src when is_binary(src) <- src,
         {:ok, t} <- Vapor.Template.compile(src) do
      {:ok, %{source: src, template: t, bos: token_str(cfg["bos_token"]), eos: token_str(cfg["eos_token"])}}
    else
      nil -> {:error, :none}
      err -> err
    end
  end

  # a list of named templates: the default one
  defp pick(l) when is_list(l), do: Enum.find_value(l, fn t -> t["name"] == "default" && t["template"] end) || (hd(l)["template"])
  defp pick(s), do: s

  defp token_str(%{"content" => c}), do: c
  defp token_str(s) when is_binary(s), do: s
  defp token_str(_), do: nil

  @doc """
  Render with a chat template (see `load_template/1`). Options: `:tools`
  (OpenAI tool definitions), `:add_generation_prompt` (default `true`),
  `:now` (the clock `strftime_now` reads), `:vars` (more template
  variables, e.g. `enable_thinking`). Messages may be maps or ordered dicts.
  Returns `{:ok, {prompt, add_bos = false, stop_ids}}` — the template writes
  the BOS it wants, as transformers' `apply_chat_template` assumes.
  """
  def render(%Tokenizer{} = tk, messages, %{template: t} = tpl, opts \\ []) do
    vars =
      %{"messages" => messages, "add_generation_prompt" => Keyword.get(opts, :add_generation_prompt, true),
        "bos_token" => tpl.bos || (tk.bos && Tokenizer.surface(tk, tk.bos)) || "",
        "eos_token" => tpl.eos || (tk.eos && Tokenizer.surface(tk, tk.eos)) || ""}
      |> then(fn v -> if tools = opts[:tools], do: Map.put(v, "tools", tools), else: v end)
      |> Map.merge(Keyword.get(opts, :vars, %{}))

    case Vapor.Template.render(t, vars, now: opts[:now]) do
      {:ok, text} -> {:ok, {text, false, stop_ids(tk)}}
      {:error, {:raised, msg}} -> {:error, "chat template: " <> msg}
      {:error, why} -> {:error, "chat template: #{inspect(why)}"}
    end
  end

  @doc "End-of-turn ids: EOS and the end-of-turn tokens the vocabulary has."
  def stop_ids(%Tokenizer{} = tk) do
    specials = Map.new(tk.special, fn {c, id, _, _} -> {c, id} end)
    ([tk.eos] ++ Enum.map(@end_of_turn, &specials[&1])) |> Enum.reject(&is_nil/1) |> Enum.uniq()
  end

  @spec render(Tokenizer.t(), [%{String.t() => String.t()}]) :: {:ok, {binary, boolean, [non_neg_integer]}} | {:error, String.t()}
  def render(%Tokenizer{} = tk, messages) when is_list(messages) do
    with :ok <- valid(messages) do
      specials = Map.new(tk.special, fn {c, id, _, _} -> {c, id} end)

      cond do
        Map.has_key?(specials, "<|im_start|>") ->
          text =
            Enum.map_join(messages, fn %{"role" => r, "content" => c} -> "<|im_start|>#{r}\n#{c}<|im_end|>\n" end) <>
              "<|im_start|>assistant\n"

          {:ok, {text, false, ids(specials, ["<|im_end|>", "<|endoftext|>"])}}

        Map.has_key?(specials, "<|start_header_id|>") ->
          text =
            "<|begin_of_text|>" <>
              Enum.map_join(messages, fn %{"role" => r, "content" => c} ->
                "<|start_header_id|>#{r}<|end_header_id|>\n\n#{String.trim(c)}<|eot_id|>"
              end) <> "<|start_header_id|>assistant<|end_header_id|>\n\n"

          {:ok, {text, false, ids(specials, ["<|eot_id|>", "<|end_of_text|>"])}}

        true ->
          {:ok, {inst(messages), true, []}}
      end
    end
  end

  defp valid(messages) do
    ok = Enum.all?(messages, &match?(%{"role" => r, "content" => c} when r in ["system", "user", "assistant"] and is_binary(c), &1))
    if ok and messages != [], do: :ok, else: {:error, "messages: a non-empty list of {role: system|user|assistant, content: string}"}
  end

  defp ids(specials, names), do: names |> Enum.map(&specials[&1]) |> Enum.reject(&is_nil/1)

  defp inst(messages) do
    {system, rest} =
      case messages do
        [%{"role" => "system", "content" => s} | rest] -> {"<<SYS>>\n#{s}\n<</SYS>>\n\n", rest}
        _ -> {"", messages}
      end

    {text, _} =
      Enum.reduce(rest, {"", system}, fn
        %{"role" => "user", "content" => c}, {acc, sys} -> {acc <> "[INST] #{sys}#{String.trim(c)} [/INST]", ""}
        %{"content" => c}, {acc, sys} -> {acc <> " #{String.trim(c)} ", sys}
      end)

    text
  end
end
