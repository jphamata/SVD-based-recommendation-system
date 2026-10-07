# Render — luz fisicamente baseada, na GPU de quem olha, com uma referência que se confere

> Pedido (0.12): "a cena e criação e edição no estúdio muito mais
> flexível e personalizável e mirando a possibilidade do foto-realismo".
> Escrutínio: [DIRETRIZ.md §15](DIRETRIZ.md).

O foto-realismo dos renderizadores de cinema vem de uma única equação —
a do transporte de luz — resolvida por amostragem de Monte Carlo
(traçado de caminhos). Aqui há dois traçadores do **mesmo formato de
cena e dos mesmos materiais**:

- `Vapor.Render` (Elixir): a **referência** — determinística por
  (semente, pixel, amostra), linhas espalhadas por todos os
  escalonadores, PNG com mapeamento ACES e sRGB;
- `priv/console/gpu_tracer.js` (WebGL2): **progressivo na GPU** do
  navegador — uma amostra por pixel por quadro acumulada em textura
  float, a imagem convergindo enquanto se olha.

Console *Fazer → Render* · MCP `render_scene`.

## 1. A cena

```
camera pos=0,1.2,4.5 look=0,0.8,0 fov=45 [aperture=… focus=…]
sky top=0.55,0.7,1.0 bottom=1,1,1          # ou color=
sun dir=0.4,1,0.3 color=1,0.95,0.85 power=2.5
plane y=0 mat=diffuse albedo=0.75,0.75,0.75 checker=0.5
sphere c=0,0.8,0 r=0.8 mat=glass ior=1.5
sphere c=-1.7,0.6,-0.5 r=0.6 mat=metal albedo=0.95,0.75,0.4 rough=0.08
box min=-0.4,0,-2 max=0.4,1.2,-1.4 mat=diffuse albedo=0.3,0.5,0.8
sphere c=0,4,0 r=0.5 mat=emit color=1,0.9,0.8 power=12
exposure value=1.2
```

Materiais: difuso de Lambert (amostragem por cosseno), metal (espelho
com rugosidade), vidro (Fresnel–Schlick, Snell, reflexão interna total),
emissivo (luzes de área). Luzes: objetos emissivos, céu (uniforme ou
gradiente vertical) e sol direcional **amostrado explicitamente**
(estimativa de evento seguinte). Roleta russa termina caminhos sem viés.

No console, o texto é a fonte da verdade e a edição é ao vivo: cada
tecla reconstrói a cena e recomeça a convergência; **arrastar a imagem
orbita a câmera e a roda aproxima — e a linha `camera` do texto é
reescrita**, de modo que o que se vê é sempre reprodutível pelo texto.

## 2. Conferido como

Do jeito que autores de renderizadores conferem os seus
(`render_test.exs`, §5h):

- **Fornalha branca**: uma esfera de albedo a num ambiente uniforme de
  radiância 1 tem de mostrar exatamente a em todo pixel — conservação de
  energia. Erro máximo 0 (a amostragem por cosseno torna cada amostra
  exata).
- **Fornalha em gradiente**: sob o céu L(ω) = (1 + ω_y)/2 uma superfície
  convexa de Lambert mostra a·(½ + n_y/3). Isso confere a *distribuição*
  do estimador, que a fornalha uniforme não vê: erro médio 6·10⁻⁴; o
  estimador viciado (direções uniformes tratadas como cosseno — o
  controle) dá a·(½ + n_y/4) e é pego (−0,04).
- **Convergência N^−½**: o erro RMS contra uma referência de 4096
  amostras cai com inclinação −0,5 ± 0,12 entre 8 e 512 amostras.
- **Determinismo**: mesma semente, mesma imagem, bit a bit.
- **A GPU contra a referência** (Chromium sem cabeça, WebGL2 sobre
  SwiftShader): a fornalha em gradiente passa na GPU e a radiância média
  de uma cena com vidro, metal e sol concorda com a do Elixir a 3 %
  (duas estimativas Monte Carlo independentes); no console, a fornalha
  branca dá concordância de 0,00 %.

## 3. Limites honestos

- Sem amostragem por importância múltipla (MIS): luzes pequenas
  alcançadas só pela BRDF — o sol visto através do vidro (cáusticas),
  emissores pequenos — têm variância de cauda pesada ("vaga-lumes").
  Por isso a conferência N^−½ usa uma cena difusa.
- Sem estrutura de aceleração: até 48 objetos na GPU e 200 no servidor;
  sem malhas de triângulos, texturas de imagem, volumes ou subsuperfície.
- O servidor limita largura × altura × amostras a 6 milhões por pedido;
  a GPU do navegador não tem esse limite.
