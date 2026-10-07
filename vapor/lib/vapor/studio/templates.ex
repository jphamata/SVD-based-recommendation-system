defmodule Vapor.Studio.Templates do
  @moduledoc """
  Starting graphs for the studio canvas (and examples for agents): each one
  runs with nothing but vapor — no uploads, no downloads — except the
  diffusion template, which needs a diffusers checkpoint directory. Node
  `ui` entries are canvas positions; the studio ignores them.
  """

  @doc "The templates: `[%{id, title, title_pt, doc, doc_pt, graph}]`."
  def all do
    [
      t("camera", "Picture to moving shot", "Imagem em plano animado",
        "A generated scene, upscaled ×2 by the consistent upscaler, then a slow push-in — a GIF whose every frame is reproducible.",
        "Uma cena gerada, ampliada ×2 pelo ampliador consistente, depois uma aproximação lenta — um GIF com cada quadro reproduzível.",
        %{"1" => node("image.scene", %{"width" => 160, "height" => 96, "seed" => 7}, %{}, 40, 60),
          "2" => node("image.upscale", %{"factor" => "2", "method" => "vapor"}, %{"image" => ["1", "image"]}, 300, 60),
          "3" => node("video.camera", %{"seconds" => 2.0, "fps" => 12.0, "zoom_start" => 1.0, "zoom_end" => 1.35}, %{"image" => ["2", "image"]}, 560, 60),
          "4" => node("studio.output", %{"name" => "shot"}, %{"value" => ["3", "video"]}, 820, 60),
          "5" => node("studio.output", %{"name" => "still"}, %{"value" => ["2", "image"]}, 560, 300)}),
      t("sound", "Sound and its picture", "Som e sua imagem",
        "A tone and seeded noise mixed, faded, and drawn as a spectrogram.",
        "Um tom e ruído com semente misturados, com fade, e desenhados como espectrograma.",
        %{"1" => node("audio.tone", %{"frequency" => 330.0, "seconds" => 1.5, "waveform" => "sine"}, %{}, 40, 40),
          "2" => node("audio.noise", %{"seconds" => 1.5, "amplitude" => 0.08, "seed" => 3}, %{}, 40, 260),
          "3" => node("audio.mix", %{"gain" => 1.0}, %{"a" => ["1", "audio"], "b" => ["2", "audio"]}, 300, 140),
          "4" => node("audio.fade", %{"fade_in" => 0.2, "fade_out" => 0.5}, %{"audio" => ["3", "audio"]}, 540, 140),
          "5" => node("audio.spectrogram", %{}, %{"audio" => ["4", "audio"]}, 780, 260),
          "6" => node("studio.output", %{"name" => "sound"}, %{"value" => ["4", "audio"]}, 780, 40),
          "7" => node("studio.output", %{"name" => "spectrogram"}, %{"value" => ["5", "image"]}, 1020, 260)}),
      t("relief", "Picture to 3D relief", "Imagem em relevo 3D",
        "A scene's luminance as a height field: a watertight mesh, rendered and turned (export GLB/OBJ/PLY from the output).",
        "A luminância de uma cena como campo de alturas: uma malha fechada, renderizada e girando (exporte GLB/OBJ/PLY na saída).",
        %{"1" => node("image.scene", %{"width" => 96, "height" => 64, "seed" => 21, "shapes" => 6}, %{}, 40, 60),
          "2" => node("image.blur", %{"radius" => 2, "sigma" => 1.2}, %{"image" => ["1", "image"]}, 290, 60),
          "3" => node("geom.heightmap", %{"height" => 0.25, "solid" => true, "max_side" => 64}, %{"image" => ["2", "image"]}, 530, 60),
          "4" => node("geom.turntable", %{"frames" => 24, "fps" => 12.0, "size" => 200}, %{"mesh" => ["3", "mesh"]}, 780, 60),
          "5" => node("studio.output", %{"name" => "mesh"}, %{"value" => ["3", "mesh"]}, 780, 300),
          "6" => node("studio.output", %{"name" => "turntable"}, %{"value" => ["4", "video"]}, 1020, 60)}),
      t("cartpole", "A trained policy and its control", "Uma política treinada e seu controle",
        "CartPole under the shipped REINFORCE policy, then under a random one: the returns are the measurement, the replay is seed + actions.",
        "CartPole sob a política REINFORCE incluída, depois sob uma aleatória: os retornos são a medida, o replay é semente + ações.",
        %{"1" => node("rl.episode", %{"env" => "cartpole", "policy" => "trained", "seed" => 11, "max_steps" => 160, "size" => 160}, %{}, 40, 40),
          "2" => node("rl.episode", %{"env" => "cartpole", "policy" => "random", "seed" => 11, "max_steps" => 160, "size" => 160}, %{}, 40, 300),
          "3" => node("video.concat", %{}, %{"a" => ["1", "video"], "b" => ["2", "video"]}, 320, 160),
          "4" => node("studio.output", %{"name" => "episodes"}, %{"value" => ["3", "video"]}, 580, 160),
          "5" => node("studio.output", %{"name" => "trained_return"}, %{"value" => ["1", "return"]}, 320, 20),
          "6" => node("studio.output", %{"name" => "random_return"}, %{"value" => ["2", "return"]}, 320, 400)}),
      t("mask", "Masked edit", "Edição com máscara",
        "Two scenes joined through a feathered rectangular mask — the compositing every inpainting starts from.",
        "Duas cenas unidas por uma máscara retangular suavizada — a composição de onde parte todo inpainting.",
        %{"1" => node("image.scene", %{"width" => 256, "height" => 160, "seed" => 4}, %{}, 40, 40),
          "2" => node("image.scene", %{"width" => 256, "height" => 160, "seed" => 99}, %{}, 40, 280),
          "3" => node("mask.rect", %{"width" => 256, "height" => 160, "x" => 64, "y" => 32, "w" => 128, "h" => 96}, %{}, 40, 520),
          "4" => node("mask.feather", %{"radius" => 8, "sigma" => 4.0}, %{"mask" => ["3", "mask"]}, 300, 520),
          "5" => node("image.composite", %{"x" => 0, "y" => 0}, %{"destination" => ["1", "image"], "source" => ["2", "image"], "mask" => ["4", "mask"]}, 560, 200),
          "6" => node("studio.output", %{"name" => "edit"}, %{"value" => ["5", "image"]}, 820, 200)}),
      t("txt2img", "Text to image (your checkpoint)", "Texto para imagem (seu checkpoint)",
        "Stable Diffusion in ComfyUI's shape: checkpoint, prompts, empty latent, sampler, VAE decode. Set the checkpoint to a diffusers directory inside the studio directory or VAPOR_MODELS.",
        "Stable Diffusion no formato do ComfyUI: checkpoint, prompts, latente vazio, amostrador, decodificação VAE. Aponte o checkpoint para um diretório diffusers dentro do diretório do estúdio ou de VAPOR_MODELS.",
        %{"1" => node("diffusion.checkpoint", %{"path" => "models/stable-diffusion"}, %{}, 40, 160),
          "2" => node("diffusion.text", %{"text" => "a lighthouse on a cliff at dawn, watercolor"}, %{"model" => ["1", "model"]}, 300, 40),
          "3" => node("diffusion.text", %{"text" => "blurry, low quality"}, %{"model" => ["1", "model"]}, 300, 260),
          "4" => node("diffusion.empty_latent", %{"width" => 512, "height" => 512}, %{}, 300, 480),
          "5" => node("diffusion.sample", %{"steps" => 25, "cfg" => 7.0, "sampler" => "dpmpp_2m", "seed" => 1},
                      %{"model" => ["1", "model"], "positive" => ["2", "conditioning"], "negative" => ["3", "conditioning"], "latent" => ["4", "latent"]}, 580, 200),
          "6" => node("diffusion.decode", %{}, %{"model" => ["1", "model"], "latent" => ["5", "latent"]}, 840, 200),
          "7" => node("studio.output", %{"name" => "image"}, %{"value" => ["6", "image"]}, 1080, 200)})
    ]
  end

  defp t(id, title, title_pt, doc, doc_pt, nodes), do: %{id: id, title: title, title_pt: title_pt, doc: doc, doc_pt: doc_pt, graph: %{"nodes" => nodes}}
  defp node(type, params, inputs, x, y), do: %{"type" => type, "params" => params, "inputs" => inputs, "ui" => %{"x" => x, "y" => y}}
end
