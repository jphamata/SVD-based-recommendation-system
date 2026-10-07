defmodule Mix.Tasks.Vapor.Generate do
  @shortdoc "Generate text from a checkpoint directory"
  @moduledoc """
      mix vapor.generate --model PATH --prompt TEXT [--max-tokens 64] [--temperature 0]
                         [--top-p 1] [--top-k 0] [--seed 0] [--threads N] [--gpu] [--quantize sb4] [--storage bf16]
                         [--max-seq 512] [--chat]

  The directory holds `config.json`, the safetensors weights and
  `tokenizer.json` or `tokenizer.gguf`. Tokens stream to stdout as they are
  produced; a summary (tokens, time, tokens/s) goes to stderr.
  """
  use Mix.Task

  @switches [model: :string, prompt: :string, max_tokens: :integer, temperature: :float, top_p: :float,
             top_k: :integer, seed: :integer, threads: :integer, gpu: :boolean, quantize: :string, storage: :string, max_seq: :integer, chat: :boolean]

  @impl true
  def run(argv) do
    Mix.Task.run("app.start")
    {o, _, _} = OptionParser.parse(argv, strict: @switches)
    dir = o[:model] || Mix.raise("--model PATH (a checkpoint directory or a .gguf file) is required")
    {:ok, tk} = Vapor.Model.tokenizer(dir) |> ok!()
    e = Vapor.CLI.engine(dir, tk, o)

    {ids, stop_ids} =
      if o[:chat] do
        {:ok, {text, bos, stop}} = Vapor.Chat.render(tk, [%{"role" => "user", "content" => Vapor.CLI.utf8_arg(o[:prompt] || "")}])
        {Vapor.Tokenizer.encode(tk, text, add_bos: bos), stop}
      else
        {Vapor.Tokenizer.encode(tk, Vapor.CLI.utf8_arg(o[:prompt] || "")), []}
      end

    t0 = System.monotonic_time()

    {:ok, ref} =
      Vapor.Engine.generate(e, ids, max_tokens: o[:max_tokens] || 64, temperature: o[:temperature] || 0.0,
                            top_p: o[:top_p], top_k: o[:top_k], seed: o[:seed], stop_ids: stop_ids)

    {n, reason} = stream(ref, "", 0)
    dt = System.convert_time_unit(System.monotonic_time() - t0, :native, :microsecond) / 1.0e6
    IO.puts(:stderr, "\n[#{length(ids)} prompt + #{n} generated tokens, #{reason}, #{Float.round(dt, 3)} s, #{Float.round(n / max(dt, 1.0e-9), 1)} tok/s]")
  end

  defp stream(ref, pending, n) do
    receive do
      {:vapor, ^ref, {:token, _, bytes}} ->
        {ready, rest} = Vapor.Serve.valid_prefix(pending <> bytes)
        IO.write(ready)
        stream(ref, rest, n + 1)

      {:vapor, ^ref, {:done, reason, _}} ->
        {n, reason}
    end
  end

  defp ok!({:ok, _} = ok), do: ok
  defp ok!({:error, r}), do: Mix.raise(inspect(r))
end
