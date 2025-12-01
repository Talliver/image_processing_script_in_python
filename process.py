"""
Comprehensive image-processing filters and glitch effects.

This module implements a large collection of pixel-manipulation filters for creative image pipelines.
The functions operate on PIL.
Image objects and return new images; 
the implementations favor clarity and portability (pure-Python pixel loops) over maximal performance.

Main effects:
- grayscale
- invert / negative
- sepia
- brightness:<factor>
- contrast:<factor>
- threshold:<level>
- posterize:<bits>
- solarize:<threshold>
- gamma:<value>
- color_balance:<r>,<g>,<b>
- blur:<radius>
- gaussian_blur:<radius>
- sharpen / unsharp:<amount>,<radius>
- emboss
- edge_detect
- edge_enhance:<amount>
- median_filter:<radius>
- add_noise:<amount>
- rotate:<degrees>
- flip_horizontal
- flip_vertical
- resize:<width>x<height>
- scale:<factor>
- hue_shift:<degrees>

Glitch / creative effects:
- rgb_offset:<offset_x>,<offset_y>
- pixel_sort:<direction>,<threshold>
- data_glitch:<intensity>
- scanline_distortion:<strength>
- channel_swap:<mode>
- vhs_glitch:<frequency>,<amount>
- chromatic_aberration:<offset>
- bit_depth_crush:<bits>
- wave_distortion:<amplitude>,<frequency>
- mosaic_glitch:<block_size>

Additional stylized filters:
- giants_causeway:<block_size>,<coolness>
- saint_remy:<swirl>,<strength>
- oil_paint:<radius>,<passes>
- watercolor:<blur_radius>,<posterize_bits>
- charcoal:<strength>
- stained_glass:<block_size>,<border>
- neon_glow:<radius>,<intensity>
- pastel_blend:<blur_radius>,<tint>
- mosaic_mural:<block_size>,<jitter>
- pointillism:<dot_size>,<density>
- film_grain:<amount>,<mono?>
- vignette:<radius>,<strength>
- fade:<amount>
- bleach_bypass:<amount>
- cross_process:<r>,<g>,<b
- scanlines:<spacing>,<intensity>
- frame_jitter:<segment>,<max_offset>
- vhs_color_bleed:<amount>
- old_film:<grain>,<scratches>,<flicker>
- dust_scratches:<density>,<size>

Usage:
- Import apply_filters and pass a PIL.Image and a list of filter spec strings:
    out = apply_filters(img, ["grayscale", "contrast:1.2", "neon_glow:8,1.2"])
- CLI example (see __main__ below) supports:
    python process.py input.jpg output.jpg --filters "grayscale,neon_glow:8,1.2"

Notes:
- Filters ignore errors and continue the pipeline where possible.
- For large images operations may be slow cause of Python.
"""

from PIL import Image, ImageDraw, ImageFilter
from typing import List, Tuple
import math
import random
from tqdm import tqdm


def _to_rgb(img: Image.Image) -> Image.Image:
	if img.mode != "RGB":
		return img.convert("RGB")
	return img


def _clamp(v: int) -> int:
	return 0 if v < 0 else (255 if v > 255 else v)


def _apply_pointwise(img: Image.Image, func) -> Image.Image:
	img = _to_rgb(img)
	pixels = list(img.getdata())
	out_pixels = [func(r, g, b) for (r, g, b) in pixels]
	out = Image.new("RGB", img.size)
	out.putdata(out_pixels)
	return out


# Grayscale: converts the image to black-and-white by luminance (desaturates).
def filter_grayscale(img: Image.Image) -> Image.Image:
	def f(r, g, b):
		y = int(0.299 * r + 0.587 * g + 0.114 * b)
		y = _clamp(y)
		return (y, y, y)

	return _apply_pointwise(img, f)


# Invert/Negative: flips colors to their opposites, producing a negative image.
def filter_invert(img: Image.Image) -> Image.Image:
	return _apply_pointwise(img, lambda r, g, b: (255 - r, 255 - g, 255 - b))


# Sepia: applies warm brownish toning for a vintage, film-like look.
def filter_sepia(img: Image.Image) -> Image.Image:
	def f(r, g, b):
		tr = int(0.393 * r + 0.769 * g + 0.189 * b)
		tg = int(0.349 * r + 0.686 * g + 0.168 * b)
		tb = int(0.272 * r + 0.534 * g + 0.131 * b)
		return (_clamp(tr), _clamp(tg), _clamp(tb))

	return _apply_pointwise(img, f)


# Brightness: scales pixel values to make the image brighter or darker.
def filter_brightness(img: Image.Image, factor: float) -> Image.Image:
	def f(r, g, b):
		return (_clamp(int(r * factor)), _clamp(int(g * factor)), _clamp(int(b * factor)))

	return _apply_pointwise(img, f)


# Contrast: increases or decreases contrast around mid-gray (128).
def filter_contrast(img: Image.Image, factor: float) -> Image.Image:
	def f(r, g, b):
		return (
			_clamp(int(128 + factor * (r - 128))),
			_clamp(int(128 + factor * (g - 128))),
			_clamp(int(128 + factor * (b - 128))),
		)

	return _apply_pointwise(img, f)


# Threshold: converts image to black-or-white based on a luminance cutoff.
def filter_threshold(img: Image.Image, level: int) -> Image.Image:
	def f(r, g, b):
		y = int(0.299 * r + 0.587 * g + 0.114 * b)
		v = 255 if y >= level else 0
		return (v, v, v)

	return _apply_pointwise(img, f)


# Posterize: reduces color depth by keeping only the most significant bits.
def filter_posterize(img: Image.Image, bits: int) -> Image.Image:
	bits = max(1, min(8, int(bits)))
	shift = 8 - bits

	def f(r, g, b):
		return ((_clamp((r >> shift) << shift)), (_clamp((g >> shift) << shift)), (_clamp((b >> shift) << shift)))

	return _apply_pointwise(img, f)


# Solarize: inverts color channels above a threshold producing a solarized look.
def filter_solarize(img: Image.Image, thresh: int) -> Image.Image:
	def f(r, g, b):
		return (r if r < thresh else 255 - r, g if g < thresh else 255 - g, b if b < thresh else 255 - b)

	return _apply_pointwise(img, f)


# Gamma: applies gamma correction to adjust midtone brightness nonlinearly.
def filter_gamma(img: Image.Image, gamma: float) -> Image.Image:
	inv = 1.0 / float(gamma)

	def f(r, g, b):
		return (
			_clamp(int(255 * ((r / 255.0) ** inv))),
			_clamp(int(255 * ((g / 255.0) ** inv))),
			_clamp(int(255 * ((b / 255.0) ** inv))),
		)

	return _apply_pointwise(img, f)


# Color balance: scales R/G/B channels to tint or correct color cast.
def filter_color_balance(img: Image.Image, rm: float, gm: float, bm: float) -> Image.Image:
	def f(r, g, b):
		return (_clamp(int(r * rm)), _clamp(int(g * gm)), _clamp(int(b * bm)))

	return _apply_pointwise(img, f)


def _convolve_rgb(img: Image.Image, kernel: List[List[int]], divisor: int = None, offset: int = 0) -> Image.Image:
	img = _to_rgb(img)
	w, h = img.size
	src = img.load()
	out = Image.new("RGB", img.size)
	dst = out.load()
	kh = len(kernel)
	kw = len(kernel[0])
	khalf_h = kh // 2
	khalf_w = kw // 2
	if divisor is None:
		divisor = sum(sum(row) for row in kernel) or 1

	for y in range(h):
		for x in range(w):
			acc_r = acc_g = acc_b = 0
			for ky in range(kh):
				for kx in range(kw):
					sx = x + (kx - khalf_w)
					sy = y + (ky - khalf_h)
					if sx < 0 or sy < 0 or sx >= w or sy >= h:
						continue
					kr = kernel[ky][kx]
					pr, pg, pb = src[sx, sy]
					acc_r += pr * kr
					acc_g += pg * kr
					acc_b += pb * kr
			r = _clamp(int(acc_r / divisor + offset))
			g = _clamp(int(acc_g / divisor + offset))
			b = _clamp(int(acc_b / divisor + offset))
			dst[x, y] = (r, g, b)
	return out


# Blur: simple normalized box blur that averages nearby pixels.
def filter_blur(img: Image.Image, radius: int = 1) -> Image.Image:
	radius = max(1, int(radius))
	size = 2 * radius + 1
	kernel = [[1 for _ in range(size)] for _ in range(size)]
	divisor = size * size
	return _convolve_rgb(img, kernel, divisor=divisor)


# Gaussian blur (approx): soft blur by repeated box blurs for a smoother look.
def filter_gaussian_blur(img: Image.Image, radius: int = 1) -> Image.Image:
	radius = max(1, int(radius))
	out = img
	for _ in range(3):
		out = filter_blur(out, radius)
	return out


# Unsharp/sharpen: enhances perceived sharpness by boosting edges from a blurred copy.
def filter_unsharp(img: Image.Image, amount: float = 1.0, radius: int = 1) -> Image.Image:
	amount = float(amount)
	radius = max(1, int(radius))
	blurred = filter_gaussian_blur(img, radius)
	img = _to_rgb(img)
	src = img.load()
	bsrc = blurred.load()
	out = Image.new("RGB", img.size)
	dst = out.load()
	w, h = img.size
	for y in range(h):
		for x in range(w):
			r = _clamp(int(src[x, y][0] + amount * (src[x, y][0] - bsrc[x, y][0])))
			g = _clamp(int(src[x, y][1] + amount * (src[x, y][1] - bsrc[x, y][1])))
			b = _clamp(int(src[x, y][2] + amount * (src[x, y][2] - bsrc[x, y][2])))
			dst[x, y] = (r, g, b)
	return out


# Emboss: gives a bas-relief effect by highlighting directional edges.
def filter_emboss(img: Image.Image) -> Image.Image:
	kernel = [[-2, -1, 0], [-1, 1, 1], [0, 1, 2]]
	return _convolve_rgb(img, kernel, divisor=1, offset=128)


# Edge detect: finds image edges and outputs their magnitude as a grayscale map.
def filter_edge_detect(img: Image.Image) -> Image.Image:
	img = _to_rgb(img)
	w, h = img.size
	src = img.load()
	out = Image.new("RGB", img.size)
	dst = out.load()
	sx = [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]
	sy = [[-1, -2, -1], [0, 0, 0], [1, 2, 1]]
	for y in range(h):
		for x in range(w):
			gx = gy = 0.0
			for ky in range(3):
				for kx in range(3):
					nx = x + (kx - 1)
					ny = y + (ky - 1)
					if nx < 0 or ny < 0 or nx >= w or ny >= h:
						continue
					r, g, b = src[nx, ny]
					lum = 0.299 * r + 0.587 * g + 0.114 * b
					gx += lum * sx[ky][kx]
					gy += lum * sy[ky][kx]
			mag = int(_clamp(math.sqrt(gx * gx + gy * gy)))
			dst[x, y] = (mag, mag, mag)
	return out


# Edge enhance: blends detected edges back into the image to increase clarity.
def filter_edge_enhance(img: Image.Image, amount: float = 1.0) -> Image.Image:
	edges = filter_edge_detect(img)
	img = _to_rgb(img)
	e = edges.load()
	s = img.load()
	out = Image.new("RGB", img.size)
	d = out.load()
	w, h = img.size
	for y in range(h):
		for x in range(w):
			er = e[x, y][0]
			d[x, y] = (
				_clamp(int(s[x, y][0] + amount * er)),
				_clamp(int(s[x, y][1] + amount * er)),
				_clamp(int(s[x, y][2] + amount * er)),
			)
	return out


# Median filter: reduces noise by replacing each pixel with the median of neighbors.
def filter_median(img: Image.Image, radius: int = 1) -> Image.Image:
	img = _to_rgb(img)
	w, h = img.size
	src = img.load()
	out = Image.new("RGB", img.size)
	dst = out.load()
	r = max(1, int(radius))
	for y in range(h):
		for x in range(w):
			rs = []
			gs = []
			bs = []
			for yy in range(max(0, y - r), min(h, y + r + 1)):
				for xx in range(max(0, x - r), min(w, x + r + 1)):
					pr, pg, pb = src[xx, yy]
					rs.append(pr)
					gs.append(pg)
					bs.append(pb)
			rs.sort()
			gs.sort()
			bs.sort()
			mid = len(rs) // 2
			dst[x, y] = (rs[mid], gs[mid], bs[mid])
	return out


# Add noise: injects random color noise (grain) into the image.
def filter_add_noise(img: Image.Image, amount: float = 0.05) -> Image.Image:
	amount = max(0.0, min(1.0, float(amount)))
	img = _to_rgb(img)
	def f(r, g, b):
		nr = _clamp(int(r + random.uniform(-1, 1) * 255 * amount))
		ng = _clamp(int(g + random.uniform(-1, 1) * 255 * amount))
		nb = _clamp(int(b + random.uniform(-1, 1) * 255 * amount))
		return (nr, ng, nb)

	return _apply_pointwise(img, f)


# Rotate: rotates the image by the given degrees (expands canvas if needed).
def filter_rotate(img: Image.Image, degrees: float) -> Image.Image:
	return img.rotate(float(degrees), expand=True)


# Flip horizontal: mirrors the image left-to-right.
def filter_flip_horizontal(img: Image.Image) -> Image.Image:
	return img.transpose(Image.FLIP_LEFT_RIGHT)


# Flip vertical: mirrors the image top-to-bottom.
def filter_flip_vertical(img: Image.Image) -> Image.Image:
	return img.transpose(Image.FLIP_TOP_BOTTOM)


# Resize: change the image dimensions to a specific width and height.
def filter_resize(img: Image.Image, width: int, height: int) -> Image.Image:
	return img.resize((int(width), int(height)))


# Scale: resize image by a multiplicative factor, preserving aspect ratio.
def filter_scale(img: Image.Image, factor: float) -> Image.Image:
	w, h = img.size
	return img.resize((max(1, int(w * float(factor))), max(1, int(h * float(factor)))))


# Hue shift: rotates hue values to change overall color balance (color wheel shift).
def filter_hue_shift(img: Image.Image, degrees: float) -> Image.Image:
	img_hsv = img.convert("HSV")
	pixels = list(img_hsv.getdata())
	shift = int((float(degrees) % 360) * 255 / 360)
	out_pixels = [((_clamp((h + shift) % 256), s, v)) for (h, s, v) in pixels]
	out = Image.new("HSV", img_hsv.size)
	out.putdata(out_pixels)
	return out.convert("RGB")


# RGB offset: shifts R/G channels differently to create a chromatic/glitch offset.
def filter_rgb_offset(img: Image.Image, offset_x: int = 2, offset_y: int = 2) -> Image.Image:
	img = _to_rgb(img)
	w, h = img.size
	src = img.load()
	out = Image.new("RGB", img.size)
	dst = out.load()
	for y in range(h):
		for x in range(w):
			r_x = (x + offset_x) % w
			g_y = (y + offset_y) % h
			r = src[r_x, y][0]
			g = src[x, g_y][1]
			b = src[x, y][2]
			dst[x, y] = (r, g, b)
	return out


# Pixel sort: reorders pixels along rows/columns by brightness to create streaks.
def filter_pixel_sort(img: Image.Image, direction: str = "horizontal", threshold: int = 128) -> Image.Image:
	img = _to_rgb(img)
	w, h = img.size
	src = img.load()
	out = Image.new("RGB", img.size)
	dst = out.load()
	direction = direction.lower()
	
	if direction == "horizontal":
		for y in range(h):
			row = [(x, src[x, y]) for x in range(w)]
			row.sort(key=lambda p: int(0.299 * p[1][0] + 0.587 * p[1][1] + 0.114 * p[1][2]))
			for idx, (_, pixel) in enumerate(row):
				dst[idx, y] = pixel
	else:
		for x in range(w):
			col = [(y, src[x, y]) for y in range(h)]
			col.sort(key=lambda p: int(0.299 * p[1][0] + 0.587 * p[1][1] + 0.114 * p[1][2]))
			for idx, (_, pixel) in enumerate(col):
				dst[x, idx] = pixel
	return out


# Data glitch: randomly replaces color channels with noise to produce glitch artifacts.
def filter_data_glitch(img: Image.Image, intensity: float = 0.1) -> Image.Image:
	intensity = max(0.0, min(1.0, float(intensity)))
	img = _to_rgb(img)
	pixels = list(img.getdata())
	out_pixels = []
	for (r, g, b) in pixels:
		if random.random() < intensity:
			r = random.randint(0, 255)
		if random.random() < intensity:
			g = random.randint(0, 255)
		if random.random() < intensity:
			b = random.randint(0, 255)
		out_pixels.append((r, g, b))
	out = Image.new("RGB", img.size)
	out.putdata(out_pixels)
	return out


# Scanline distortion: vertically offsets scanlines for a rolling/glitchy look.
def filter_scanline_distortion(img: Image.Image, strength: int = 3) -> Image.Image:
	img = _to_rgb(img)
	w, h = img.size
	src = img.load()
	out = Image.new("RGB", img.size)
	dst = out.load()
	strength = max(1, int(strength))
	
	for y in range(h):
		offset = random.randint(-strength, strength)
		src_y = (y + offset) % h
		for x in range(w):
			dst[x, y] = src[x, src_y]
	return out


# Channel swap: permutes R/G/B channels to change colors (e.g., RGB->BGR).
def filter_channel_swap(img: Image.Image, mode: str = "bgr") -> Image.Image:
	img = _to_rgb(img)
	mode = mode.lower().strip()
	def f(r, g, b):
		if mode == "bgr":
			return (b, g, r)
		elif mode == "grb":
			return (g, r, b)
		elif mode == "rbg":
			return (r, b, g)
		elif mode == "brg":
			return (b, r, g)
		elif mode == "gbr":
			return (g, b, r)
		else:
			return (r, g, b)
	return _apply_pointwise(img, f)


# VHS glitch: intermittently shifts horizontal bands to mimic VHS tape artifacts.
def filter_vhs_glitch(img: Image.Image, frequency: int = 5, amount: int = 8) -> Image.Image:
	img = _to_rgb(img)
	w, h = img.size
	src = img.load()
	out = Image.new("RGB", img.size)
	dst = out.load()
	frequency = max(1, int(frequency))
	amount = max(1, int(amount))
	
	for y in range(h):
		if (y // frequency) % 2 == 0:
			offset = random.randint(-amount, amount)
			for x in range(w):
				src_x = (x + offset) % w
				dst[x, y] = src[src_x, y]
		else:
			for x in range(w):
				dst[x, y] = src[x, y]
	return out


# Chromatic aberration: offsets color channels to create fringing at edges.
def filter_chromatic_aberration(img: Image.Image, offset: int = 3) -> Image.Image:
	img = _to_rgb(img)
	w, h = img.size
	src = img.load()
	out = Image.new("RGB", img.size)
	dst = out.load()
	offset = max(1, int(offset))
	
	for y in range(h):
		for x in range(w):
			r_x = min(w - 1, x + offset)
			g_y = min(h - 1, y + offset)
			b_x = max(0, x - offset)
			r = src[r_x, y][0]
			g = src[x, g_y][1]
			b = src[b_x, y][2]
			dst[x, y] = (r, g, b)
	return out


# Film grain: adds random grain; mono makes grain luminance-only.
def filter_film_grain(img: Image.Image, amount: float = 0.08, mono: bool = False) -> Image.Image:
	amount = max(0.0, min(1.0, float(amount)))
	img = _to_rgb(img)
	def f(r, g, b):
		if mono:
			n = int(random.uniform(-1, 1) * 255 * amount)
			return (_clamp(r + n), _clamp(g + n), _clamp(b + n))
		else:
			nr = int(random.uniform(-1, 1) * 255 * amount)
			ng = int(random.uniform(-1, 1) * 255 * amount)
			nb = int(random.uniform(-1, 1) * 255 * amount)
			return (_clamp(r + nr), _clamp(g + ng), _clamp(b + nb))
	return _apply_pointwise(img, f)


# Vignette: darkens edges progressively for a focused center look.
def filter_vignette(img: Image.Image, radius: float = 0.75, strength: float = 0.6) -> Image.Image:
	img = _to_rgb(img)
	w, h = img.size
	cx = w / 2.0
	cy = h / 2.0
	max_dist = math.hypot(cx, cy) * (radius if radius > 0 else 1.0)
	src = img.load()
	out = Image.new("RGB", img.size)
	dst = out.load()
	for y in range(h):
		for x in range(w):
			dx = x - cx
			dy = y - cy
			d = math.hypot(dx, dy)
			factor = 1.0 - (d / max_dist)
			factor = max(0.0, min(1.0, factor))
			vign = (factor ** 2) * (1.0 - strength) + strength * 0.0
			r, g, b = src[x, y]
			r = int(r * (vign + (1.0 - strength)))
			g = int(g * (vign + (1.0 - strength)))
			b = int(b * (vign + (1.0 - strength)))
			dst[x, y] = (_clamp(r), _clamp(g), _clamp(b))
	return out


# Fade: desaturates and slightly alters contrast/brightness for a faded look.
def filter_fade(img: Image.Image, amount: float = 0.25) -> Image.Image:
	amount = max(0.0, min(1.0, float(amount)))
	img = _to_rgb(img)
	desat = filter_grayscale(img)
	out = Image.blend(img, desat, amount * 0.7)
	out = filter_contrast(out, 1.0 - amount * 0.2)
	out = filter_brightness(out, 1.0 + amount * 0.05)
	return out


# Bleach bypass: high-contrast, desaturated filmic look by blending with grayscale.
def filter_bleach_bypass(img: Image.Image, amount: float = 0.8) -> Image.Image:
	amount = max(0.0, min(1.0, float(amount)))
	img = _to_rgb(img)
	desat = filter_grayscale(img)
	out = Image.blend(img, desat, 0.6 * amount)
	out = filter_contrast(out, 1.1 + amount * 0.6)
	out = filter_color_balance(out, 1.0 - amount * 0.05, 1.0 - amount * 0.05, 1.0 + amount * 0.02)
	return out


# Cross process: shifts color channels to imitate cross-processed film chemistry.
def filter_cross_process(img: Image.Image, r: int = 10, g: int = -6, b: int = 6) -> Image.Image:
	img = _to_rgb(img)
	def f(rr, gg, bb):
		nr = _clamp(int(rr + r))
		ng = _clamp(int(gg + g))
		nb = _clamp(int(bb + b))
		nr = _clamp(int(128 + 1.05 * (nr - 128)))
		ng = _clamp(int(128 + 1.05 * (ng - 128)))
		nb = _clamp(int(128 + 1.05 * (nb - 128)))
		return (nr, ng, nb)
	return _apply_pointwise(img, f)


# Scanlines: overlays horizontal scanlines to mimic old CRT/VHS displays.
def filter_scanlines(img: Image.Image, spacing: int = 3, intensity: float = 0.12) -> Image.Image:
	img = _to_rgb(img)
	w, h = img.size
	out = img.copy()
	draw = ImageDraw.Draw(out)
	spacing = max(1, int(spacing))
	intensity = max(0.0, min(1.0, float(intensity)))
	for y in range(0, h, spacing):
		draw.line([(0, y), (w, y)], fill=(int(0), int(0), int(0)), width=1)
	out = Image.blend(img, out, intensity)
	return out


# Frame jitter: breaks image into horizontal segments and offsets them for jitter.
def filter_frame_jitter(img: Image.Image, segment: int = 20, max_offset: int = 6) -> Image.Image:
	img = _to_rgb(img)
	w, h = img.size
	out = Image.new("RGB", img.size)
	src = img.load()
	dst = out.load()
	seg = max(1, int(segment))
	maxo = max(0, int(max_offset))
	for by in range(0, h, seg):
		offset = random.randint(-maxo, maxo)
		for y in range(by, min(h, by + seg)):
			for x in range(w):
				sx = (x + offset) % w
				dst[x, y] = src[sx, y]
	return out


# VHS color bleed: shifts and blurs color channels to create color bleeding.
def filter_vhs_color_bleed(img: Image.Image, amount: int = 6) -> Image.Image:
	img = _to_rgb(img)
	w, h = img.size
	r = img.copy()
	g = img.copy()
	b = img.copy()
	r = r.transform((w, h), Image.AFFINE, (1, 0, amount/2.0, 0, 1, 0))
	b = b.transform((w, h), Image.AFFINE, (1, 0, -amount/2.0, 0, 1, 0))
	r = r.filter(ImageFilter.GaussianBlur(max(1, amount//3)))
	b = b.filter(ImageFilter.GaussianBlur(max(1, amount//3)))
	rr = r.split()[0]
	gg = g.split()[1]
	bb = b.split()[2]
	out = Image.merge('RGB', (rr, gg, bb))
	return out


# Old film: combines grain, scratches, flicker, and sepia for vintage film effect.
def filter_old_film(img: Image.Image, grain: float = 0.08, scratches: float = 0.002, flicker: float = 0.06) -> Image.Image:
	img = _to_rgb(img)
	out = filter_film_grain(img, grain, mono=False)
	w, h = out.size
	draw = ImageDraw.Draw(out)
	density = max(0.0, min(1.0, float(scratches)))
	area = w * h
	count = int(area * density)
	for _ in range(count):
		x = random.randint(0, w - 1)
		y = random.randint(0, h - 1)
		size = random.randint(1, 3)
		col = (255, 255, 255) if random.random() < 0.6 else (0, 0, 0)
		draw.ellipse([x, y, min(w - 1, x + size), min(h - 1, y + size)], fill=col)
	band = max(1, int(h * 0.02))
	src = out.load()
	for y in range(0, h, band):
		factor = 1.0 - (random.uniform(-1, 1) * flicker)
		for yy in range(y, min(h, y + band)):
			for x in range(w):
				r, g, b = src[x, yy]
				src[x, yy] = (_clamp(int(r * factor)), _clamp(int(g * factor)), _clamp(int(b * factor)))
	out = filter_sepia(out)
	return out


# Dust & scratches: paints random specks and small rectangles to simulate damage.
def filter_dust_scratches(img: Image.Image, density: float = 0.001, size: int = 2) -> Image.Image:
	img = _to_rgb(img)
	out = img.copy()
	w, h = out.size
	draw = ImageDraw.Draw(out)
	density = max(0.0, min(1.0, float(density)))
	area = w * h
	count = int(area * density)
	for _ in range(count):
		x = random.randint(0, w - 1)
		y = random.randint(0, h - 1)
		s = max(1, int(size))
		col = (255, 255, 255) if random.random() < 0.7 else (0, 0, 0)
		draw.rectangle([x, y, min(w - 1, x + s), min(h - 1, y + s)], fill=col)
	return out


# Bit-depth crush: reduces color precision and adds small quantization noise.
def filter_bit_depth_crush(img: Image.Image, bits: int = 2) -> Image.Image:
	bits = max(1, min(8, int(bits)))
	shift = 8 - bits
	img = _to_rgb(img)
	def f(r, g, b):
		noise = random.randint(-2, 2)
		return (
			_clamp(((r >> shift) << shift) + noise),
			_clamp(((g >> shift) << shift) + noise),
			_clamp(((b >> shift) << shift) + noise),
		)
	return _apply_pointwise(img, f)


# Wave distortion: displaces pixels with sinusoidal offsets for rippling effect.
def filter_wave_distortion(img: Image.Image, amplitude: int = 5, frequency: float = 0.05) -> Image.Image:
	img = _to_rgb(img)
	w, h = img.size
	src = img.load()
	out = Image.new("RGB", img.size)
	dst = out.load()
	amplitude = max(1, int(amplitude))
	frequency = max(0.01, float(frequency))
	
	for y in range(h):
		for x in range(w):
			wave_x = int(amplitude * math.sin(y * frequency))
			wave_y = int(amplitude * math.cos(x * frequency))
			src_x = (x + wave_x) % w
			src_y = (y + wave_y) % h
			dst[x, y] = src[src_x, src_y]
	return out


# Mosaic glitch: randomly shifts small blocks to create pixelated glitch patches.
def filter_mosaic_glitch(img: Image.Image, block_size: int = 8) -> Image.Image:
	img = _to_rgb(img)
	w, h = img.size
	src = img.load()
	out = Image.new("RGB", img.size)
	dst = out.load()
	block_size = max(1, int(block_size))
	
	for y in range(h):
		for x in range(w):
			dst[x, y] = src[x, y]
	
	for by in range(0, h, block_size):
		for bx in range(0, w, block_size):
			if random.random() < 0.3:
				offset_x = random.randint(-block_size * 2, block_size * 2)
				offset_y = random.randint(-block_size * 2, block_size * 2)
				for dy in range(min(block_size, h - by)):
					for dx in range(min(block_size, w - bx)):
						src_x = (bx + dx + offset_x) % w
						src_y = (by + dy + offset_y) % h
						dst[bx + dx, by + dy] = src[src_x, src_y]
	return out


# Giants Causeway: stylized block-averaged mosaic with edge blending for texture.
def filter_giants_causeway(img: Image.Image, block_size: int = 12, coolness: float = 0.2) -> Image.Image:
	img = _to_rgb(img)
	w, h = img.size
	src = img.load()
	out = Image.new("RGB", img.size)
	dst = out.load()
	bs = max(2, int(block_size))
	for by in range(0, h, bs):
		for bx in range(0, w, bs):
			r_total = g_total = b_total = count = 0
			for y in range(by, min(h, by + bs)):
				for x in range(bx, min(w, bx + bs)):
					r, g, b = src[x, y]
					r_total += r; g_total += g; b_total += b; count += 1
			if count == 0:
				continue
			r_avg = int(r_total / count)
			g_avg = int(g_total / count)
			b_avg = int(b_total / count)
			r_avg = _clamp(int(r_avg * (1.0 - coolness)))
			b_avg = _clamp(int(b_avg * (1.0 + coolness)))
			for y in range(by, min(h, by + bs)):
				for x in range(bx, min(w, bx + bs)):
					dst[x, y] = (r_avg, g_avg, b_avg)
	ed = filter_edge_detect(img)
	ed = ed.point(lambda p: int(p * 0.6))
	out = Image.blend(out, ed, 0.12)
	return out


# Saint Remy: painterly stylization combining swirl, hue shift, and posterization.
def filter_saint_remy(img: Image.Image, swirl: int = 6, strength: float = 1.2) -> Image.Image:
	out = filter_wave_distortion(img, amplitude=swirl, frequency=0.03)
	out = filter_hue_shift(out, 12)
	out = filter_unsharp(out, amount=strength, radius=1)
	out = filter_bit_depth_crush(out, 6)
	return out


# Oil paint: smoothing and posterizing to simulate thick paint strokes.
def filter_oil_paint(img: Image.Image, radius: int = 3, passes: int = 2) -> Image.Image:
	out = img.copy()
	for _ in range(max(1, int(passes))):
		out = filter_median(out, radius)
	out = filter_posterize(out, max(2, 8 - radius))
	return out


# Watercolor: soft blur with edge blend to create a painted, watercolor look.
def filter_watercolor(img: Image.Image, blur_radius: int = 2, posterize_bits: int = 5) -> Image.Image:
	out = filter_gaussian_blur(img, int(blur_radius))
	out = filter_posterize(out, int(posterize_bits))
	ed = filter_edge_detect(img)
	ed = ed.point(lambda p: 255 - p)
	out = Image.blend(out, ed.convert("RGB"), 0.15)
	return out


# Charcoal: high-contrast grayscale with edge ink-like strokes.
def filter_charcoal(img: Image.Image, strength: float = 1.5) -> Image.Image:
	gl = filter_grayscale(img)
	ed = filter_edge_detect(img)
	ed = ed.point(lambda p: _clamp(int(255 - p * 1.2)))
	out = Image.blend(gl, ed.convert("RGB"), 0.6)
	out = filter_contrast(out, strength)
	return out


# Stained glass: block-averaged tiles with dark borders like leaded glass.
def filter_stained_glass(img: Image.Image, block_size: int = 16, border: int = 2) -> Image.Image:
	img = _to_rgb(img)
	w, h = img.size
	out = Image.new("RGB", img.size)
	draw = ImageDraw.Draw(out)
	bs = max(2, int(block_size))
	for by in range(0, h, bs):
		for bx in range(0, w, bs):
			r_total = g_total = b_total = count = 0
			for y in range(by, min(h, by + bs)):
				for x in range(bx, min(w, bx + bs)):
					r, g, b = img.getpixel((x, y))
					r_total += r; g_total += g; b_total += b; count += 1
			if count == 0:
				continue
			r_avg = int(r_total / count)
			g_avg = int(g_total / count)
			b_avg = int(b_total / count)
			draw.rectangle([bx, by, bx + bs - 1, by + bs - 1], fill=(r_avg, g_avg, b_avg))
			draw.rectangle([bx, by, bx + bs - 1, by + bs - 1], outline=(30, 30, 30), width=max(1, int(border)))
	return out


# Neon glow: brightens high-luminance areas and blurs them to create a glow.
def filter_neon_glow(img: Image.Image, radius: int = 8, intensity: float = 1.2) -> Image.Image:
	img = _to_rgb(img)
	w, h = img.size
	lum = img.convert("L")
	mask = lum.point(lambda p: 255 if p > 180 else 0)
	glow = img.copy().filter(ImageFilter.GaussianBlur(radius))
	glow = filter_brightness(glow, intensity)
	out = Image.composite(glow, img, mask.convert("L"))
	return out


# Pastel blend: soft blur with a subtle color tint for pastel tones.
def filter_pastel_blend(img: Image.Image, blur_radius: int = 4, tint: str = "#FFD1DC") -> Image.Image:
	out = img.filter(ImageFilter.GaussianBlur(blur_radius))
	tint_img = Image.new("RGB", out.size, tint)
	out = Image.blend(out, tint_img, 0.12)
	out = filter_contrast(out, 0.9)
	return out


# Mosaic mural: large block mosaic with jitter to create an abstract mural effect.
def filter_mosaic_mural(img: Image.Image, block_size: int = 30, jitter: int = 8) -> Image.Image:
	img = _to_rgb(img)
	w, h = img.size
	out = Image.new("RGB", img.size)
	dst = out.load()
	bs = max(2, int(block_size))
	for by in range(0, h, bs):
		for bx in range(0, w, bs):
			dx = bx + random.randint(-jitter, jitter)
			dy = by + random.randint(-jitter, jitter)
			for y in range(by, min(h, by + bs)):
				for x in range(bx, min(w, bx + bs)):
					sx = min(max(0, dx + (x - bx)), w - 1)
					sy = min(max(0, dy + (y - by)), h - 1)
					dst[x, y] = img.getpixel((sx, sy))
	out = filter_contrast(out, 1.1)
	return out


# Pointillism: paints colored dots sampled from the image to emulate pointillist art.
def filter_pointillism(img: Image.Image, dot_size: int = 3, density: float = 0.03) -> Image.Image:
	img = _to_rgb(img)
	w, h = img.size
	out = Image.new("RGB", img.size, (255, 255, 255))
	draw = ImageDraw.Draw(out)
	ds = max(1, int(dot_size))
	for y in range(0, h):
		for x in range(0, w):
			if random.random() < density:
				r, g, b = img.getpixel((x, y))
				draw.ellipse([x, y, x + ds, y + ds], fill=(r, g, b))
	return out


def parse_filter_name(spec: str) -> Tuple[str, str]:
	if ":" in spec:
		name, arg = spec.split(":", 1)
		return name.strip().lower(), arg.strip()
	return spec.strip().lower(), ""


def apply_filters(img: Image.Image, filters: List[str]) -> Image.Image:
	out = img.copy()
	# show progress across the list of filters for better visibility
	for spec in tqdm(filters, desc="Applying filters", unit="filter"):
		name, arg = parse_filter_name(spec)
		try:
			if name == "grayscale":
				out = filter_grayscale(out)
			elif name == "invert" or name == "negative":
				out = filter_invert(out)
			elif name == "sepia":
				out = filter_sepia(out)
			elif name == "brightness":
				out = filter_brightness(out, float(arg or 1.0))
			elif name == "contrast":
				out = filter_contrast(out, float(arg or 1.0))
			elif name == "threshold":
				out = filter_threshold(out, int(arg or 128))
			elif name == "posterize":
				out = filter_posterize(out, int(arg or 4))
			elif name == "solarize":
				out = filter_solarize(out, int(arg or 128))
			elif name == "gamma":
				out = filter_gamma(out, float(arg or 1.0))
			elif name == "color_balance":
				parts = [float(x) for x in (arg.split(",") if arg else ["1","1","1"])]
				while len(parts) < 3:
					parts.append(1.0)
				out = filter_color_balance(out, parts[0], parts[1], parts[2])
			elif name == "blur":
				out = filter_blur(out, int(arg or 1))
			elif name == "gaussian_blur":
				out = filter_gaussian_blur(out, int(arg or 1))
			elif name == "sharpen" or name == "unsharp":
				parts = arg.split(",") if arg else ["1.0","1"]
				amt = float(parts[0])
				radius = int(parts[1]) if len(parts) > 1 else 1
				out = filter_unsharp(out, amt, radius)
			elif name == "emboss":
				out = filter_emboss(out)
			elif name == "edge_detect":
				out = filter_edge_detect(out)
			elif name == "edge_enhance":
				out = filter_edge_enhance(out, float(arg or 1.0))
			elif name == "median_filter":
				out = filter_median(out, int(arg or 1))
			elif name == "add_noise":
				out = filter_add_noise(out, float(arg or 0.05))
			elif name == "rotate":
				out = filter_rotate(out, float(arg or 0.0))
			elif name == "flip_horizontal":
				out = filter_flip_horizontal(out)
			elif name == "flip_vertical":
				out = filter_flip_vertical(out)
			elif name == "resize":
				if "x" in arg:
					w, h = arg.split("x", 1)
					out = filter_resize(out, int(w), int(h))
			elif name == "scale":
				out = filter_scale(out, float(arg or 1.0))
			elif name == "hue_shift":
				out = filter_hue_shift(out, float(arg or 0.0))
			elif name == "rgb_offset":
				parts = [int(x) for x in (arg.split(",") if arg else ["2","2"])]
				ox = parts[0] if len(parts) > 0 else 2
				oy = parts[1] if len(parts) > 1 else 2
				out = filter_rgb_offset(out, ox, oy)
			elif name == "pixel_sort":
				parts = arg.split(",") if arg else ["horizontal","128"]
				direction = parts[0].strip() if len(parts) > 0 else "horizontal"
				threshold = int(parts[1]) if len(parts) > 1 else 128
				out = filter_pixel_sort(out, direction, threshold)
			elif name == "data_glitch":
				out = filter_data_glitch(out, float(arg or 0.1))
			elif name == "scanline_distortion":
				out = filter_scanline_distortion(out, int(arg or 3))
			elif name == "channel_swap":
				out = filter_channel_swap(out, arg or "bgr")
			elif name == "vhs_glitch":
				parts = [int(x) for x in (arg.split(",") if arg else ["5","8"])]
				freq = parts[0] if len(parts) > 0 else 5
				amt = parts[1] if len(parts) > 1 else 8
				out = filter_vhs_glitch(out, freq, amt)
			elif name == "film_grain":
				parts = arg.split(",") if arg else ["0.08","False"]
				amt = float(parts[0]) if parts[0] else 0.08
				mono = parts[1].lower() in ("1","true","t","yes","y") if len(parts) > 1 else False
				out = filter_film_grain(out, amt, mono)
			elif name == "vignette":
				parts = arg.split(",") if arg else ["0.75","0.6"]
				out = filter_vignette(out, float(parts[0]), float(parts[1]) if len(parts) > 1 else float(parts[1]))
			elif name == "fade":
				out = filter_fade(out, float(arg or 0.25))
			elif name == "bleach_bypass":
				out = filter_bleach_bypass(out, float(arg or 0.8))
			elif name == "cross_process":
				parts = [int(x) for x in (arg.split(",") if arg else ["10","-6","6"])]
				while len(parts) < 3:
					parts.append(0)
				out = filter_cross_process(out, parts[0], parts[1], parts[2])
			elif name == "scanlines":
				parts = arg.split(",") if arg else ["3","0.12"]
				out = filter_scanlines(out, int(parts[0]), float(parts[1]))
			elif name == "frame_jitter":
				parts = arg.split(",") if arg else ["20","6"]
				out = filter_frame_jitter(out, int(parts[0]), int(parts[1]))
			elif name == "vhs_color_bleed":
				out = filter_vhs_color_bleed(out, int(arg or 6))
			elif name == "old_film":
				parts = arg.split(",") if arg else ["0.08","0.002","0.06"]
				out = filter_old_film(out, float(parts[0]), float(parts[1]), float(parts[2]))
			elif name == "dust_scratches":
				parts = arg.split(",") if arg else ["0.001","2"]
				out = filter_dust_scratches(out, float(parts[0]), int(parts[1]))
			elif name == "chromatic_aberration":
				out = filter_chromatic_aberration(out, int(arg or 3))
			elif name == "bit_depth_crush":
				out = filter_bit_depth_crush(out, int(arg or 2))
			elif name == "wave_distortion":
				parts = arg.split(",") if arg else ["5","0.05"]
				amp = int(parts[0]) if len(parts) > 0 else 5
				freq = float(parts[1]) if len(parts) > 1 else 0.05
				out = filter_wave_distortion(out, amp, freq)
			elif name == "mosaic_glitch":
				out = filter_mosaic_glitch(out, int(arg or 8))
			elif name == "giants_causeway":
				parts = arg.split(",") if arg else ["12","0.2"]
				bs = int(parts[0]) if parts[0] else 12
				cool = float(parts[1]) if len(parts) > 1 else 0.2
				out = filter_giants_causeway(out, bs, cool)
			elif name == "saint_remy":
				parts = arg.split(",") if arg else ["6","1.2"]
				sw = int(parts[0])
				strg = float(parts[1])
				out = filter_saint_remy(out, sw, strg)
			elif name == "oil_paint":
				parts = arg.split(",") if arg else ["3","2"]
				r = int(parts[0])
				p = int(parts[1])
				out = filter_oil_paint(out, r, p)
			elif name == "watercolor":
				parts = arg.split(",") if arg else ["2","5"]
				out = filter_watercolor(out, int(parts[0]), int(parts[1]))
			elif name == "charcoal":
				out = filter_charcoal(out, float(arg or 1.5))
			elif name == "stained_glass":
				parts = arg.split(",") if arg else ["16","2"]
				out = filter_stained_glass(out, int(parts[0]), int(parts[1]))
			elif name == "neon_glow":
				parts = arg.split(",") if arg else ["8","1.2"]
				out = filter_neon_glow(out, int(parts[0]), float(parts[1]))
			elif name == "pastel_blend":
				parts = arg.split(",") if arg else ["4","#FFD1DC"]
				out = filter_pastel_blend(out, int(parts[0]), parts[1])
			elif name == "mosaic_mural":
				parts = arg.split(",") if arg else ["30","8"]
				out = filter_mosaic_mural(out, int(parts[0]), int(parts[1]))
			elif name == "pointillism":
				parts = arg.split(",") if arg else ["3","0.03"]
				out = filter_pointillism(out, int(parts[0]), float(parts[1]))
			else:
				continue
		except Exception:
			continue
	return out


# Demonstration / CLI usage
def _demo_apply(input_path: str, filters_list, output_path: str):
	try:
		img = Image.open(input_path)
	except Exception as e:
		print(f"Failed to open '{input_path}': {e}")
		return
	out = apply_filters(img, filters_list)
	try:
		out.save(output_path)
		print(f"Saved result to '{output_path}' (filters: {filters_list})")
	except Exception as e:
		print(f"Failed to save '{output_path}': {e}")


def _demo_presets(input_path: str, out_dir: str):
	import os
	os.makedirs(out_dir, exist_ok=True)
	presets = {
		"neon": ["contrast:1.2", "neon_glow:8,1.4"],
		"glitch": ["rgb_offset:6,0", "scanline_distortion:4", "data_glitch:0.06"],
		"painterly": ["watercolor:2,5", "oil_paint:3,2"],
		"vintage": ["old_film:0.06,0.001,0.06", "vignette:0.8,0.6"],
		"vhs": ["vhs_glitch:6,10", "vhs_color_bleed:8", "scanlines:3,0.12"],
	}
	for name, filt in presets.items():
		out_path = os.path.join(out_dir, f"{name}.jpg")
		_demo_apply(input_path, filt, out_path)


if __name__ == "__main__":
	import argparse
	parser = argparse.ArgumentParser(description="Apply filters from process.py")
	parser.add_argument("input", help="Input image path")
	parser.add_argument("output", help="Output image path (or directory when --presets)")
	parser.add_argument("--filters", default="", help='Comma-separated filters, e.g. "grayscale,contrast:1.2"')
	parser.add_argument("--presets", action="store_true", help="Generate a few preset outputs into the output dir")
	args = parser.parse_args()

	if args.presets:
		_demo_presets(args.input, args.output)
	else:
		if args.filters.strip() == "":
			print("No filters specified; nothing to do.")
		else:
			filters = [s.strip() for s in args.filters.split(",") if s.strip()]
			_demo_apply(args.input, filters, args.output)
