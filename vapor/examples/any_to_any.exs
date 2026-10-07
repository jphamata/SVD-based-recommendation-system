# Any-to-any in the world of Vapor.Modal.World: fit every codec in closed
# form, route between text, image and audio, and write what comes out.
#
#     mix run examples/any_to_any.exs [OUT_DIR]
#
# Every step is a program built through the model airlock and run on the
# native worker when there is one (the oracle otherwise) — same bits.
alias Vapor.Modal.{Audio, Hub, Image, Runner, World}
alias Vapor.Quality.Signal

out = List.first(System.argv()) || Path.join(System.tmp_dir!(), "vapor-any-to-any")
File.mkdir_p!(out)
w = Runner.worker()
IO.puts("fitting the hub on #{if w, do: "the native worker", else: "the oracle"} …")
hub = Hub.fit_world(worker: w)
IO.puts("routes: #{Hub.routes(hub) |> Enum.map_join(", ", fn {a, b} -> "#{a}→#{b}" end)}")

# a scene no fit has seen
img = World.scene({"blue", "yellow"}, 4242)
Image.write_png(Path.join(out, "input.png"), img, 8)

{:ok, caption, _} = Hub.convert(hub, :image, :text_colour, img)
{:ok, back, _} = Hub.convert(hub, :text_colour, :image, caption)
{:ok, sound, route} = Hub.convert(hub, :image, :audio, img)
{:ok, vq, _} = Hub.convert(hub, :image, :image, img)
{:ok, soft, _} = Hub.convert(hub, :image, :text_note_soft, img)

Image.write_png(Path.join(out, "caption_to_image.png"), back, 8)
Image.write_png(Path.join(out, "vq_round_trip.png"), vq, 8)
Audio.write(Path.join(out, "image_to_audio.wav"), sound)

IO.puts("""
image → text:            #{Enum.join(caption, " ")}
text → image:            PSNR #{Float.round(Signal.psnr(World.scene({"blue", "yellow"}), back), 1)} dB vs the canonical scene
image → image (VQ):      PSNR #{Float.round(Signal.psnr(img, vq), 1)} dB
image → audio:           #{Float.round(Signal.dominant_hz(sound), 1)} Hz (the note of "blue" is #{World.hz(World.note_of("blue"))} Hz), route #{inspect(route)}
image → soft token → LM: #{Enum.join(soft, " ")}
wrote #{out}/
""")
