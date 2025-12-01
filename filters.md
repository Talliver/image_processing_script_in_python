# Available Filters

This document lists every filter implemented in `process.py`, with a short description, parameters (with defaults), and a CLI usage example.

Basic adjustments

- **grayscale**: Convert image to luminance-based grayscale. Example: `--filters "grayscale"`.
- **invert / negative**: Invert all color channels (negative). Example: `--filters "invert"`.
- **sepia**: Apply warm, brownish toning. Example: `--filters "sepia"`.
- **brightness**: Scale brightness by factor. Params: `(factor: float, default=1.0)`. Example: `--filters "brightness:1.2"`.
- **contrast**: Scale contrast about mid-gray. Params: `(factor: float, default=1.0)`. Example: `--filters "contrast:1.3"`.
- **threshold**: Binarize by luminance cutoff. Params: `(level: int, default=128)`. Example: `--filters "threshold:150"`.
- **posterize**: Reduce color bit-depth. Params: `(bits: int, 1-8, default=4)`. Example: `--filters "posterize:3"`.
- **solarize**: Invert channels above a threshold. Params: `(threshold: int, default=128)`. Example: `--filters "solarize:120"`.
- **gamma**: Gamma-correct image. Params: `(gamma: float, default=1.0)`. Example: `--filters "gamma:0.8"`.
- **color_balance**: Scale R,G,B channels independently. Params: `(r:float,g:float,b:float, default=1,1,1)`. Example: `--filters "color_balance:1.1,0.95,0.9"`.

Convolution & spatial

- **blur**: Box blur radius. Params: `(radius: int, default=1)`. Example: `--filters "blur:2"`.
- **gaussian_blur**: Approximated Gaussian by repeated box blurs. Params: `(radius: int, default=1)`. Example: `--filters "gaussian_blur:3"`.
- **sharpen / unsharp**: Unsharp mask style sharpening. Params: `(amount:float, radius:int)`, default `1.0,1`. Example: `--filters "unsharp:1.2,2"`.
- **emboss**: Bas-relief embossing. Example: `--filters "emboss"`.
- **edge_detect**: Sobel-like edge magnitude (grayscale). Example: `--filters "edge_detect"`.
- **edge_enhance**: Blend edges into image. Params: `(amount: float, default=1.0)`. Example: `--filters "edge_enhance:1.5"`.
- **median_filter**: Median neighborhood filter. Params: `(radius: int, default=1)`. Example: `--filters "median_filter:2"`.

Noise & glitch

- **add_noise**: Add random color noise. Params: `(amount: float 0..1, default=0.05)`. Example: `--filters "add_noise:0.08"`.
- **data_glitch**: Randomly replace channels with noise. Params: `(intensity: float 0..1, default=0.1)`. Example: `--filters "data_glitch:0.12"`.
- **film_grain**: Add grain (optionally mono). Params: `(amount:float, mono:bool, default=0.08,False)`. Example: `--filters "film_grain:0.12,True"`.
- **bit_depth_crush**: Reduce bit depth (quantize + small noise). Params: `(bits:int, 1-8, default=2)`. Example: `--filters "bit_depth_crush:3"`.
- **add_noise** (alias): see above.

Color/channel manipulations

- **rgb_offset**: Shift red/green channels separately. Params: `(offset_x:int, offset_y:int, default=2,2)`. Example: `--filters "rgb_offset:6,0"`.
- **channel_swap**: Permute R/G/B order. Params: `(mode:str, default="bgr")` modes include `bgr, grb, rbg, brg, gbr`. Example: `--filters "channel_swap:grb"`.
- **hue_shift**: Rotate hue degrees. Params: `(degrees:float, default=0)`. Example: `--filters "hue_shift:45"`.
- **chromatic_aberration**: Fringe channels by offset. Params: `(offset:int, default=3)`. Example: `--filters "chromatic_aberration:4"`.
- **vhs_color_bleed**: Shift/blur channels to create bleed. Params: `(amount:int, default=6)`. Example: `--filters "vhs_color_bleed:8"`.
- **cross_process**: Shift channels to simulate cross-processed film. Params: `(r:int,g:int,b:int, default=10,-6,6)`. Example: `--filters "cross_process:12,-4,8"`.

Geometric / transforms

- **rotate**: Rotate image (expand canvas). Params: `(degrees:float, default=0)`. Example: `--filters "rotate:90"`.
- **flip_horizontal**: Mirror left-to-right. Example: `--filters "flip_horizontal"`.
- **flip_vertical**: Mirror top-to-bottom. Example: `--filters "flip_vertical"`.
- **resize**: Resize to exact `widthxheight`. Params: `(width x height)`. Example: `--filters "resize:800x600"`.
- **scale**: Scale by factor (preserves aspect). Params: `(factor:float, default=1.0)`. Example: `--filters "scale:0.5"`.

Distortions & scanline effects

- **scanline_distortion**: Randomly offset scanlines. Params: `(strength:int, default=3)`. Example: `--filters "scanline_distortion:4"`.
- **vhs_glitch**: Intermittent band shifts to mimic VHS. Params: `(frequency:int,amount:int, default=5,8)`. Example: `--filters "vhs_glitch:6,10"`.
- **scanlines**: Overlay horizontal scanlines. Params: `(spacing:int,intensity:float, default=3,0.12)`. Example: `--filters "scanlines:4,0.1"`.
- **frame_jitter**: Break image into horizontal segments and jitter them. Params: `(segment:int,max_offset:int, default=20,6)`. Example: `--filters "frame_jitter:30,10"`.
- **wave_distortion**: Sinusoidal pixel displacement. Params: `(amplitude:int,frequency:float, default=5,0.05)`. Example: `--filters "wave_distortion:8,0.08"`.
- **mosaic_glitch**: Randomly shift small blocks (pixelated glitch). Params: `(block_size:int, default=8)`. Example: `--filters "mosaic_glitch:12"`.

Sorting & painterly glitches

- **pixel_sort**: Reorder pixels along rows/columns by brightness. Params: `(direction:str,threshold:int, default="horizontal",128)`. Example: `--filters "pixel_sort:vertical,100"`.
- **giants_causeway**: Block-averaged mosaic with edge blending. Params: `(block_size:int,coolness:float, default=12,0.2)`. Example: `--filters "giants_causeway:16,0.3"`.
- **mosaic_mural**: Large-block mosaic with jitter. Params: `(block_size:int,jitter:int, default=30,8)`. Example: `--filters "mosaic_mural:40,12"`.

Painterly / stylized effects

- **oil_paint**: Smooth + posterize to emulate oil strokes. Params: `(radius:int,passes:int, default=3,2)`. Example: `--filters "oil_paint:4,3"`.
- **watercolor**: Blur + posterize + edge blend. Params: `(blur_radius:int,posterize_bits:int, default=2,5)`. Example: `--filters "watercolor:3,6"`.
- **saint_remy**: Composite painterly preset (swirl, hue shift, posterize). Params: `(swirl:int,strength:float, default=6,1.2)`. Example: `--filters "saint_remy:8,1.4"`.
- **neon_glow**: Brighten high-luminance areas and blur for glow. Params: `(radius:int,intensity:float, default=8,1.2)`. Example: `--filters "neon_glow:10,1.4"`.
- **pastel_blend**: Soft blur + color tint. Params: `(blur_radius:int,tint:hex, default=4,#FFD1DC)`. Example: `--filters "pastel_blend:6,#E8D4FF"`.
- **pointillism**: Paint colored dots sampled from image. Params: `(dot_size:int,density:float, default=3,0.03)`. Example: `--filters "pointillism:4,0.05"`.
- **charcoal**: High-contrast grayscale with edge strokes. Params: `(strength:float, default=1.5)`. Example: `--filters "charcoal:2.0"`.
- **stained_glass**: Block tiles with dark borders. Params: `(block_size:int,border:int, default=16,2)`. Example: `--filters "stained_glass:20,3"`.

Vintage & film effects

- **old_film**: Grain + scratches + flicker + sepia. Params: `(grain:float,scratches:float,flicker:float, default=0.08,0.002,0.06)`. Example: `--filters "old_film:0.06,0.001,0.05"`.
- **dust_scratches**: Paint specks/rectangles for damage. Params: `(density:float,size:int, default=0.001,2)`. Example: `--filters "dust_scratches:0.002,3"`.
- **vignette**: Darken edges to focus center. Params: `(radius:float,strength:float, default=0.75,0.6)`. Example: `--filters "vignette:0.8,0.7"`.
- **fade**: Desaturate + slight contrast/brightness changes. Params: `(amount:float, default=0.25)`. Example: `--filters "fade:0.4"`.
- **bleach_bypass**: High-contrast desaturated filmic look. Params: `(amount:float, default=0.8)`. Example: `--filters "bleach_bypass:0.9"`.

Utility / compose

- **resize** / **scale** / **rotate** / **flip_horizontal** / **flip_vertical**: geometric helpers (see above). Example: `--filters "resize:1024x768,rotate:90,flip_horizontal"`.

CLI usage notes

- Multiple filters are applied in sequence. Example:

```bash
python process.py input.jpg output.jpg --filters "grayscale,contrast:1.2,neon_glow:8,1.2"
```

- Filters accept comma-separated arguments after `:` when required. If a parameter is omitted, the filter uses its default.

If you want me to expand any filter entry with more detail (algorithm notes, performance tips, or parameter ranges), tell me which filters to expand.
