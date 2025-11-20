"""Comprehensive set of manually-coded filters for the image pipeline.

This module implements 20+ filters without relying on PIL's ImageFilter
implementations. The filters operate on pixel data (getdata/putdata or
neighborhood kernels) and return a new PIL Image. The `apply_filters`
function accepts a list of string names; a filter may accept a parameter
after a colon (e.g. 'brightness:1.2', 'resize:200x300').

Supported filters (examples / parameter forms):
- grayscale
- invert
- sepia
- brightness:<factor>         # e.g. 1.2
- contrast:<factor>           # e.g. 1.3 (1.0 = no change)
- threshold:<level>           # 0-255
- posterize:<bits>            # 1-8
- solarize:<threshold>        # 0-255
- gamma:<value>               # >0
- color_balance:<r>,<g>,<b>   # multipliers
- blur:<radius>               # box blur
- gaussian_blur:<radius>      # approximated via repeated box blur
- sharpen:<amount>            # unsharp mask style amount
- emboss                       # emboss kernel
- edge_detect                  # sobel-like magnitude
- edge_enhance:<amount>        # add edges to original
- median_filter:<radius>       # median/denoise
- add_noise:<amount>           # amount 0-1 (fraction of max noise)
- rotate:<degrees>
- flip_horizontal
- flip_vertical
- resize:<width>x<height> or scale:<factor>
- hue_shift:<degrees>

The implementations favor clarity over extreme performance (pure-Python
pixel loops). For large images some filters may be slow; consider adding
NumPy-based variants if needed.
"""

from PIL import Image
from typing import List, Tuple
import math
import random


def _to_rgb(img: Image.Image) -> Image.Image:
	if img.mode != "RGB":
		return img.convert("RGB")
	return img


def _clamp(v: int) -> int:
	return 0 if v < 0 else (255 if v > 255 else v)


def _apply_pointwise(img: Image.Image, func) -> Image.Image:
	"""Apply `func(r,g,b) -> (r,g,b)` to each pixel and return new image."""
	img = _to_rgb(img)
	pixels = list(img.getdata())
	out_pixels = [func(r, g, b) for (r, g, b) in pixels]
	out = Image.new("RGB", img.size)
	out.putdata(out_pixels)
	return out


def filter_grayscale(img: Image.Image) -> Image.Image:
	def f(r, g, b):
		y = int(0.299 * r + 0.587 * g + 0.114 * b)
		y = _clamp(y)
		return (y, y, y)

	return _apply_pointwise(img, f)


def filter_invert(img: Image.Image) -> Image.Image:
	return _apply_pointwise(img, lambda r, g, b: (255 - r, 255 - g, 255 - b))


def filter_sepia(img: Image.Image) -> Image.Image:
	def f(r, g, b):
		tr = int(0.393 * r + 0.769 * g + 0.189 * b)
		tg = int(0.349 * r + 0.686 * g + 0.168 * b)
		tb = int(0.272 * r + 0.534 * g + 0.131 * b)
		return (_clamp(tr), _clamp(tg), _clamp(tb))

	return _apply_pointwise(img, f)


def filter_brightness(img: Image.Image, factor: float) -> Image.Image:
	def f(r, g, b):
		return (_clamp(int(r * factor)), _clamp(int(g * factor)), _clamp(int(b * factor)))

	return _apply_pointwise(img, f)


def filter_contrast(img: Image.Image, factor: float) -> Image.Image:
	# simple contrast around midpoint 128
	def f(r, g, b):
		return (
			_clamp(int(128 + factor * (r - 128))),
			_clamp(int(128 + factor * (g - 128))),
			_clamp(int(128 + factor * (b - 128))),
		)

	return _apply_pointwise(img, f)


def filter_threshold(img: Image.Image, level: int) -> Image.Image:
	def f(r, g, b):
		y = int(0.299 * r + 0.587 * g + 0.114 * b)
		v = 255 if y >= level else 0
		return (v, v, v)

	return _apply_pointwise(img, f)


def filter_posterize(img: Image.Image, bits: int) -> Image.Image:
	bits = max(1, min(8, int(bits)))
	shift = 8 - bits

	def f(r, g, b):
		return ((_clamp((r >> shift) << shift)), (_clamp((g >> shift) << shift)), (_clamp((b >> shift) << shift)))

	return _apply_pointwise(img, f)


def filter_solarize(img: Image.Image, thresh: int) -> Image.Image:
	def f(r, g, b):
		return (r if r < thresh else 255 - r, g if g < thresh else 255 - g, b if b < thresh else 255 - b)

	return _apply_pointwise(img, f)


def filter_gamma(img: Image.Image, gamma: float) -> Image.Image:
	inv = 1.0 / float(gamma)

	def f(r, g, b):
		return (
			_clamp(int(255 * ((r / 255.0) ** inv))),
			_clamp(int(255 * ((g / 255.0) ** inv))),
			_clamp(int(255 * ((b / 255.0) ** inv))),
		)

	return _apply_pointwise(img, f)


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


def filter_blur(img: Image.Image, radius: int = 1) -> Image.Image:
	# box blur kernel
	radius = max(1, int(radius))
	size = 2 * radius + 1
	kernel = [[1 for _ in range(size)] for _ in range(size)]
	divisor = size * size
	return _convolve_rgb(img, kernel, divisor=divisor)


def filter_gaussian_blur(img: Image.Image, radius: int = 1) -> Image.Image:
	# approximate gaussian by applying box blur multiple times
	radius = max(1, int(radius))
	out = img
	# three passes approximates a gaussian for small radii
	for _ in range(3):
		out = filter_blur(out, radius)
	return out


def filter_unsharp(img: Image.Image, amount: float = 1.0, radius: int = 1) -> Image.Image:
	# Unsharp mask: out = original + amount * (original - blurred)
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


def filter_emboss(img: Image.Image) -> Image.Image:
	kernel = [[-2, -1, 0], [-1, 1, 1], [0, 1, 2]]
	return _convolve_rgb(img, kernel, divisor=1, offset=128)


def filter_edge_detect(img: Image.Image) -> Image.Image:
	# simple Sobel operator magnitude on luminance then map to RGB
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


def filter_add_noise(img: Image.Image, amount: float = 0.05) -> Image.Image:
	amount = max(0.0, min(1.0, float(amount)))
	img = _to_rgb(img)
	def f(r, g, b):
		nr = _clamp(int(r + random.uniform(-1, 1) * 255 * amount))
		ng = _clamp(int(g + random.uniform(-1, 1) * 255 * amount))
		nb = _clamp(int(b + random.uniform(-1, 1) * 255 * amount))
		return (nr, ng, nb)

	return _apply_pointwise(img, f)


def filter_rotate(img: Image.Image, degrees: float) -> Image.Image:
	return img.rotate(float(degrees), expand=True)


def filter_flip_horizontal(img: Image.Image) -> Image.Image:
	return img.transpose(Image.FLIP_LEFT_RIGHT)


def filter_flip_vertical(img: Image.Image) -> Image.Image:
	return img.transpose(Image.FLIP_TOP_BOTTOM)


def filter_resize(img: Image.Image, width: int, height: int) -> Image.Image:
	return img.resize((int(width), int(height)))


def filter_scale(img: Image.Image, factor: float) -> Image.Image:
	w, h = img.size
	return img.resize((max(1, int(w * float(factor))), max(1, int(h * float(factor)))))


def filter_hue_shift(img: Image.Image, degrees: float) -> Image.Image:
	# operate in HSV space using pillow's convert; H is 0-255 in Pillow
	img_hsv = img.convert("HSV")
	pixels = list(img_hsv.getdata())
	shift = int((float(degrees) % 360) * 255 / 360)
	out_pixels = [((_clamp((h + shift) % 256), s, v)) for (h, s, v) in pixels]
	out = Image.new("HSV", img_hsv.size)
	out.putdata(out_pixels)
	return out.convert("RGB")


def parse_filter_name(spec: str) -> Tuple[str, str]:
	if ":" in spec:
		name, arg = spec.split(":", 1)
		return name.strip().lower(), arg.strip()
	return spec.strip().lower(), ""


def apply_filters(img: Image.Image, filters: List[str]) -> Image.Image:
	"""Apply a list of filters (strings) in order and return resulting image.

	Each filter may optionally include a parameter after a colon, for example
	'brightness:1.2' or 'resize:200x300'. Unknown filters are ignored.
	"""
	out = img.copy()
	for spec in filters:
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
			else:
				# unknown filter name: ignore
				continue
		except Exception:
			# ignore filter errors and continue with pipeline
			continue
	return out
