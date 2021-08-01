from colour import Color
import numpy as np
import random
from functools import partial

# def color_rgb_scale(a: Color, b: Color, n:int=10):
#     """Interpolate between two colors on rgb scale"""
#     argb = np.array(a.rgb)
#     brgb = np.array(b.rgb)
#     d = (brgb-argb)
#     cs = argb[:, None]+ np.array([np.linspace(0, dd, n) for dd in d])
#     return [Color(rgb=cc) for cc in cs.T]

def color_rgb_rand(a:Color, b:Color, n:int=10):
    """Gen rand colors along rgb diff between two colors."""
    argb = np.array(a.rgb)
    brgb = np.array(b.rgb)
    d = (brgb - argb)
    r = np.random.rand(3, n) * d[:, None]
    r = np.sort(r)
    cs = argb[:, None]+ r
    return [Color(rgb=cc) for cc in cs.T]

# # random green color
# def rand_color_tween_two_of(colors: list):
#     """Choses two color and interpolates random color between them."""
#     a = np.random.choice(colors)
#     b = np.random.choice(colors)
#     c = color_rgb_rand(a, b, 1)[0].rgb
#     return c2rgba(c)

def c2rgba(color):
    return (*color.rgb, 1)
    
def gen_rand_color_between_two_of(colors: list, n=50):
    """Choses two colors, return func that gives random color between them"""
    a, b = np.random.choice(colors, 2, replace=False)
    colors = color_rgb_rand(a, b, n)
    colors_rgba = [c2rgba(c) for c in colors]
    return partial(random.choice, colors_rgba)
