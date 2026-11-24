"""YANI logo: a sibling of the YAMC mark, with capture instead of a track.

YAMC shows a particle traversing the letters, which is what a transport code
does. yani transmutes, so the motif is a full reaction: a neutron arrives from
the left and is absorbed into a large nuclide sitting on the seam between "A"
and "N", the colour boundary the wordmark already has. The compound nuclide
splits into an alpha particle, which drops away below it, and a recoiling
daughter, which is what emits the gamma continuing on through "N" and "I".
Letters keep their normal kerning -- the motif is sized to read clearly at
letter height, which means it rides over the top of "A" and "N" the way
YAMC's own track rides over its letters, rather than being shrunk to fit the
sliver of whitespace between them. Bound nucleons are drawn in their own
physical colours, blue neutrons and red protons, rather than the green used
for a neutron in flight; that green, and the letters' blue and orange, are
sampled from the YAMC png.
"""
import math
from PIL import Image, ImageDraw, ImageFont

BLUE = (22, 84, 127, 255)
ORANGE = (238, 118, 27, 255)
GREEN = (73, 150, 89, 255)
RED = (196, 58, 51, 255)
WHITE = (255, 255, 255, 255)
# A lighter blue for neutrons: the letters already own BLUE, and a nucleon in
# that exact shade disappears where the motif overlaps "A".
NEUTRON = (64, 148, 199, 255)

SS = 4                                  # supersample, then downsample to smooth
W, H = 627 * SS, 225 * SS
im = Image.new("RGBA", (W, H), (0, 0, 0, 0))
d = ImageDraw.Draw(im)

# Size the wordmark to fill the canvas, the way YAMC's does, rather than
# leaving a third of it empty. Letters keep their normal kerning throughout --
# nothing here spreads them apart to make room for the motif.
PAD = 16 * SS
size = 150 * SS
while True:
    f = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", size)
    if d.textlength("YANI", font=f) >= W - 2 * PAD or size > 400 * SS:
        break
    size += 4 * SS
font = f
box = d.textbbox((0, 0), "YANI", font=font)
text_y = (H - (box[3] - box[1])) // 2 - box[1]

# --- letters: YA blue, NI orange, matching YAMC's split -----------------------
x = PAD
for chars, colour in (("YA", BLUE), ("NI", ORANGE)):
    d.text((x, text_y), chars, font=font, fill=colour)
    x += d.textlength(chars, font=font)
total_w = x - PAD
an_seam = PAD + d.textlength("YA", font=font)   # x where "A" ends and "N" begins


def disc(cx, cy, r, fill):
    d.ellipse([cx - r, cy - r, cx + r, cy + r], fill=fill)


def nucleon_cluster(cx, cy, positions, colours, nucleon_r):
    for (ox, oy), colour in zip(positions, colours):
        disc(cx + ox, cy + oy, nucleon_r, colour)


# --- the reaction sits on the A/N seam, at letter height -- lower than a
# separate band above the wordmark, and without spreading the letters apart
# to clear room for it --------------------------------------------------------
nx, ny = int(an_seam), int(H * 0.36)
start = (int(W * 0.03), int(H * 0.92))

# incoming neutron, dashed
dx, dy = nx - start[0], ny - start[1]
length = math.hypot(dx, dy)
ux, uy = dx / length, dy / length
travel = length - 56 * SS
seg, gap, pos = 26 * SS, 18 * SS, 0
while pos < travel:
    e = min(pos + seg, travel)
    seg_pts = [start[0] + ux * pos, start[1] + uy * pos,
               start[0] + ux * e, start[1] + uy * e]
    d.line(seg_pts, fill=GREEN, width=8 * SS)
    pos += seg + gap
disc(start[0] + ux * (travel + 14 * SS), start[1] + uy * (travel + 14 * SS),
     11 * SS, GREEN)

# the large nuclide the neutron is absorbed into: enough nucleons, in two
# rings around a centre one, to read as "large" next to the four-nucleon
# alpha below it. Physical colours: blue neutrons, red protons.
big_positions = [(0, 0)]
for a in (i * math.pi / 3 for i in range(6)):
    big_positions.append((15 * SS * math.cos(a), 15 * SS * math.sin(a)))
for a in (i * math.pi / 4 + 0.4 for i in range(8)):
    big_positions.append((28 * SS * math.cos(a), 28 * SS * math.sin(a)))
big_colours = [NEUTRON if i % 2 == 0 else RED for i in range(len(big_positions))]
nucleon_cluster(nx, ny, big_positions, big_colours, nucleon_r=9 * SS)

# the alpha particle (2 protons, 2 neutrons), dropped well clear of the
# nuclide so the two read as separate, with an arrow between them showing the
# alpha being ejected from the product rather than just a floating line
ax_, ay_ = nx - 2 * SS, ny + 92 * SS
arrow_from = (nx, ny + 30 * SS)
arrow_to = (ax_, ay_ - 26 * SS)
d.line([*arrow_from, *arrow_to], fill=GREEN, width=6 * SS)
aang = math.atan2(arrow_to[1] - arrow_from[1], arrow_to[0] - arrow_from[0])
ahead = [
    (arrow_to[0] + 15 * SS * math.cos(aang), arrow_to[1] + 15 * SS * math.sin(aang)),
    (arrow_to[0] + 12 * SS * math.cos(aang + 2.5), arrow_to[1] + 12 * SS * math.sin(aang + 2.5)),
    (arrow_to[0] + 12 * SS * math.cos(aang - 2.5), arrow_to[1] + 12 * SS * math.sin(aang - 2.5)),
]
d.polygon(ahead, fill=GREEN)
alpha_positions = [(-9 * SS, -9 * SS), (9 * SS, -9 * SS), (-9 * SS, 9 * SS), (9 * SS, 9 * SS)]
alpha_colours = [RED, RED, NEUTRON, NEUTRON]
nucleon_cluster(ax_, ay_, alpha_positions, alpha_colours, nucleon_r=9 * SS)

# --- outgoing gamma from the recoiling daughter, continuing on through N and I
gx, gy = nx + 18 * SS, ny - 12 * SS
pts = []
for k in range(110):
    t = k / 109
    px = gx + t * (W * 0.30)
    py = gy - t * 30 * SS + math.sin(t * math.pi * 3.0) * 10 * SS
    pts.append((px, py))
d.line(pts, fill=BLUE, width=8 * SS, joint="curve")
(px0, py0), (px1, py1) = pts[-10], pts[-1]
ang = math.atan2(py1 - py0, px1 - px0)
head = [
    (px1 + 24 * SS * math.cos(ang), py1 + 24 * SS * math.sin(ang)),
    (px1 + 11 * SS * math.cos(ang + 2.5), py1 + 11 * SS * math.sin(ang + 2.5)),
    (px1 + 11 * SS * math.cos(ang - 2.5), py1 + 11 * SS * math.sin(ang - 2.5)),
]
d.polygon(head, fill=BLUE)

im = im.resize((627, 225), Image.LANCZOS)
im.save("logo.png")
print("wrote logo.png", im.size, "letters width", total_w // SS)

# A square favicon crop: the nucleus alone, which is the distinctive part.
fav = Image.new("RGBA", (512, 512), (0, 0, 0, 0))
fd = ImageDraw.Draw(fav)
cx = cy = 256
for k in range(6):
    a = math.pi / 3 * k + 0.4
    r = 108
    fd.ellipse([cx + r * math.cos(a) - 74, cy + r * math.sin(a) - 74,
                cx + r * math.cos(a) + 74, cy + r * math.sin(a) + 74],
               fill=NEUTRON if k % 2 == 0 else RED)
fd.ellipse([cx - 80, cy - 80, cx + 80, cy + 80], fill=BLUE)
fav.save("favicon.png")
print("wrote favicon.png")
