"""Огурчик-философ: генерирует видео с говорящим огурцом.
Требуется: python3, Pillow, numpy, ffmpeg, espeak-ng.
Запуск: python3 make_video.py [out.mp4]
"""
import math, os, subprocess, sys, tempfile, wave
import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

W, H, FPS, S = 1280, 720, 30, 2  # S = supersampling
FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
FONT_B = "/usr/share/fonts/truetype/dejavu/DejaVuSerif-Bold.ttf"

LINES = [
    "Здравствуйте. Я огурчик. И сегодня я хочу поговорить о философии.",
    "Сократ говорил: я знаю, что ничего не знаю. Я тоже. Я же огурец.",
    "Декарт сказал: мыслю, следовательно, существую. А я хрущу, следовательно, существую.",
    "Гераклит учил, что нельзя дважды войти в одну и ту же реку. А в одну и ту же банку можно. Проверено.",
    "Экзистенциалисты спрашивают: в чём смысл жизни? Может быть, в том, чтобы стать солёным, но остаться собой.",
    "Камю писал, что Сизифа нужно представлять счастливым. А меня представляйте хрустящим.",
    "Помните: все мы немного огурцы в банке бытия. Спасибо за внимание.",
]
PAUSE = 0.6
SR = 22050


def tts(text, path):
    subprocess.run(["espeak-ng", "-v", "ru", "-s", "135", "-p", "75", "-a", "180",
                    "-w", path, text], check=True)
    with wave.open(path) as w:
        sr = w.getframerate()
        a = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16).astype(np.float32)
    if sr != SR:
        a = np.interp(np.linspace(0, len(a), int(len(a) * SR / sr), endpoint=False), np.arange(len(a)), a)
    return a


def build_audio(tmp):
    chunks, segs, t = [np.zeros(int(SR * 0.8))], [], 0.8
    for i, line in enumerate(LINES):
        a = tts(line, os.path.join(tmp, f"l{i}.wav"))
        segs.append((t, t + len(a) / SR, line))
        chunks += [a, np.zeros(int(SR * PAUSE))]
        t += len(a) / SR + PAUSE
    chunks.append(np.zeros(int(SR * 1.2)))
    audio = np.concatenate(chunks)
    audio = audio / (np.abs(audio).max() + 1e-6) * 0.9
    path = os.path.join(tmp, "voice.wav")
    with wave.open(path, "wb") as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(SR)
        w.writeframes((audio * 32767).astype(np.int16).tobytes())
    # mouth envelope per video frame
    n = int(len(audio) / SR * FPS) + 1
    hop = SR // FPS
    env = np.array([np.sqrt(np.mean(audio[i*hop:(i+1)*hop] ** 2)) if i*hop < len(audio) else 0 for i in range(n)])
    env = np.clip(env / (np.percentile(env[env > 0.01], 90) + 1e-6), 0, 1)
    env = np.convolve(env, [0.25, 0.5, 0.25], mode="same")
    return path, segs, env, n


def background():
    g = np.linspace(0, 1, H * S)[:, None]
    top, bot = np.array([34, 24, 52]), np.array([14, 30, 40])
    img = (top * (1 - g) + bot * g)[:, :, None].repeat(W * S, 1).transpose(0, 1, 2)
    img = np.repeat(((top * (1 - g) + bot * g))[:, None, :], W * S, axis=1).astype(np.uint8)
    bg = Image.fromarray(img, "RGB")
    d = ImageDraw.Draw(bg)
    # spotlight
    spot = Image.new("L", bg.size, 0)
    ImageDraw.Draw(spot).ellipse([W*S*0.25, H*S*0.05, W*S*0.75, H*S*1.1], fill=70)
    spot = spot.filter(ImageFilter.GaussianBlur(120))
    bg = Image.composite(Image.new("RGB", bg.size, (90, 80, 120)), bg, spot)
    d = ImageDraw.Draw(bg)
    # floor / table
    d.rectangle([0, H*S*0.82, W*S, H*S], fill=(60, 38, 30))
    d.rectangle([0, H*S*0.82, W*S, H*S*0.835], fill=(90, 58, 42))
    # books on the left
    x = 80 * S
    for c, h in [((150, 40, 40), 90), ((40, 80, 140), 110), ((180, 140, 50), 80), ((60, 120, 70), 100)]:
        d.rectangle([x, H*S*0.82 - h*S, x + 28*S, H*S*0.82], fill=c)
        d.rectangle([x + 4*S, H*S*0.82 - h*S + 12*S, x + 24*S, H*S*0.82 - h*S + 16*S], fill=(230, 210, 160))
        x += 32 * S
    # pickle jar on the right
    jx, jy = W*S - 230*S, H*S*0.82
    d.rounded_rectangle([jx, jy - 170*S, jx + 120*S, jy], 20*S, fill=(120, 170, 150), outline=(200, 230, 220), width=4*S)
    d.rectangle([jx - 6*S, jy - 190*S, jx + 126*S, jy - 165*S], fill=(170, 60, 50))
    for k in range(3):
        d.rounded_rectangle([jx + 15*S + k*33*S, jy - 150*S, jx + 40*S + k*33*S, jy - 15*S], 12*S, fill=(70, 130, 60))
    f = ImageFont.truetype(FONT_B, 34 * S)
    d.text((W*S/2, 40*S), "Огурчик-философ", font=f, fill=(235, 225, 190), anchor="mt")
    return bg


def cucumber_body():
    bw, bh = 300 * S, 470 * S
    im = Image.new("RGBA", (bw + 80*S, bh + 40*S), (0, 0, 0, 0))
    d = ImageDraw.Draw(im)
    ox, oy = 40 * S, 20 * S
    d.rounded_rectangle([ox + 8*S, oy + 10*S, ox + bw + 8*S, oy + bh + 10*S], bw // 2, fill=(20, 50, 20, 120))
    d.rounded_rectangle([ox, oy, ox + bw, oy + bh], bw // 2, fill=(70, 150, 60))
    # light stripes
    for k, sx in enumerate([0.25, 0.5, 0.75]):
        x = ox + bw * sx
        d.rounded_rectangle([x - 14*S, oy + 40*S, x + 14*S, oy + bh - 40*S], 14*S, fill=(95, 175, 80))
    # highlight
    d.rounded_rectangle([ox + 40*S, oy + 60*S, ox + 70*S, oy + 220*S], 15*S, fill=(150, 210, 130))
    # bumps
    rng = np.random.default_rng(3)
    for _ in range(38):
        x = ox + rng.uniform(0.12, 0.88) * bw
        y = oy + rng.uniform(0.08, 0.92) * bh
        if 0.2 * bh < y - oy < 0.62 * bh and 0.15 * bw < x - ox < 0.85 * bw:
            continue  # keep face clean
        r = rng.uniform(4, 7) * S
        d.ellipse([x - r, y - r, x + r, y + r], fill=(45, 110, 40))
        d.ellipse([x - r*0.4, y - r*0.6, x + r*0.2, y - r*0.1], fill=(140, 200, 120))
    # stem
    d.rounded_rectangle([ox + bw/2 - 12*S, oy - 18*S, ox + bw/2 + 12*S, oy + 10*S], 8*S, fill=(110, 90, 40))
    # cheeks
    for cx in (0.22, 0.78):
        c = Image.new("RGBA", im.size, (0, 0, 0, 0))
        ImageDraw.Draw(c).ellipse([ox + bw*cx - 30*S, oy + bh*0.47 - 16*S, ox + bw*cx + 30*S, oy + bh*0.47 + 16*S], fill=(240, 120, 120, 110))
        im = Image.alpha_composite(im, c.filter(ImageFilter.GaussianBlur(8*S)))
    return im, (ox, oy, bw, bh)


def draw_face(d, x0, y0, bw, bh, t, mouth, blink, talking):
    cx = x0 + bw / 2
    ey = y0 + bh * 0.32
    # eyebrows (raise when talking loud)
    lift = 8 * S * mouth + 3 * S * math.sin(t * 1.3)
    for sgn in (-1, 1):
        ex = cx + sgn * 62 * S
        d.line([ex - 32*S, ey - 50*S - lift + sgn*4*S, ex + 32*S, ey - 50*S - lift - sgn*4*S], fill=(30, 50, 25), width=9*S)
    # eyes
    look = 6 * S * math.sin(t * 0.7)
    for sgn in (-1, 1):
        ex = cx + sgn * 62 * S
        h = 34 * S * (1 - blink) + 3 * S
        d.ellipse([ex - 30*S, ey - h, ex + 30*S, ey + h], fill=(250, 250, 245), outline=(30, 50, 25), width=4*S)
        if blink < 0.7:
            pr = 15 * S
            px, py = ex + look, ey + 4*S
            d.ellipse([px - pr, py - min(pr, h), px + pr, py + min(pr, h)], fill=(30, 25, 25))
            d.ellipse([px - 7*S, py - 9*S, px - 1*S, py - 3*S], fill=(255, 255, 255))
    # round philosopher glasses
    for sgn in (-1, 1):
        ex = cx + sgn * 62 * S
        d.ellipse([ex - 44*S, ey - 44*S, ex + 44*S, ey + 44*S], outline=(40, 30, 20), width=6*S)
    d.arc([cx - 20*S, ey - 18*S, cx + 20*S, ey + 10*S], 200, 340, fill=(40, 30, 20), width=6*S)
    # mouth
    my = y0 + bh * 0.53
    mw = 52 * S + 10 * S * mouth
    mh = 6 * S + 46 * S * mouth
    if mh < 10 * S:
        d.arc([cx - mw, my - 30*S, cx + mw, my + 14*S], 20, 160, fill=(40, 25, 25), width=7*S)
    else:
        d.ellipse([cx - mw*0.8, my - mh*0.35, cx + mw*0.8, my + mh], fill=(90, 25, 35), outline=(40, 25, 25), width=5*S)
        d.ellipse([cx - mw*0.45, my + mh*0.45, cx + mw*0.45, my + mh*0.95], fill=(220, 100, 110))
    # beard (wise)
    by = y0 + bh * 0.66
    pts = [(cx - 70*S, by), (cx + 70*S, by), (cx + 40*S, by + 60*S), (cx, by + 100*S + 6*S*math.sin(t*3)), (cx - 40*S, by + 60*S)]
    d.polygon(pts, fill=(235, 235, 225))
    for k in range(-2, 3):
        d.line([cx + k*22*S, by + 10*S, cx + k*12*S, by + 70*S], fill=(200, 200, 190), width=3*S)


def wrap(text, font, maxw, d):
    words, lines, cur = text.split(), [], ""
    for w in words:
        trial = (cur + " " + w).strip()
        if d.textlength(trial, font=font) <= maxw:
            cur = trial
        else:
            lines.append(cur); cur = w
    lines.append(cur)
    return lines


def main():
    out = sys.argv[1] if len(sys.argv) > 1 else "cucumber_philosopher.mp4"
    with tempfile.TemporaryDirectory() as tmp:
        wav, segs, env, n = build_audio(tmp)
        bg = background()
        body, (ox, oy, bw, bh) = cucumber_body()
        sub_font = ImageFont.truetype(FONT, 30)
        silent = os.path.join(tmp, "video.mp4")
        ff = subprocess.Popen(["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
                               "-s", f"{W}x{H}", "-r", str(FPS), "-i", "-", "-i", wav,
                               "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "20", "-preset", "medium",
                               "-c:a", "aac", "-b:a", "128k", "-shortest", out], stdin=subprocess.PIPE)
        blinks = set()
        rng = np.random.default_rng(7)
        f = 20
        while f < n:
            blinks.update(range(f, f + 6)); f += int(rng.uniform(70, 140))
        for i in range(n):
            t = i / FPS
            m = float(env[i])
            talking = any(a <= t <= b for a, b, _ in segs)
            frame = bg.copy()
            bob = int((6 * math.sin(t * 2.2) + 10 * m) * S)
            sway = int(10 * S * math.sin(t * 0.9))
            px = int(W * S / 2 - body.width / 2) + sway
            py = int(H * S * 0.82 - body.height + 10 * S) - bob
            frame.paste(body, (px, py), body)
            d = ImageDraw.Draw(frame)
            blink = 0.0
            if i in blinks:
                k = i - min(b for b in blinks if b <= i and i - b < 6)
                blink = [0.4, 0.9, 1.0, 0.9, 0.5, 0.1][k]
            draw_face(d, px + ox, py + oy, bw, bh, t, m if talking else 0.0, blink, talking)
            frame = frame.resize((W, H), Image.LANCZOS)
            # subtitles
            cur = next((s for s in segs if s[0] - 0.1 <= t <= s[1] + PAUSE * 0.8), None)
            if cur:
                d2 = ImageDraw.Draw(frame, "RGBA")
                lines = wrap(cur[2], sub_font, W - 220, d2)
                lh = 40
                top = H - 30 - lh * len(lines)
                d2.rounded_rectangle([90, top - 14, W - 90, H - 18], 16, fill=(0, 0, 0, 170))
                for k, ln in enumerate(lines):
                    d2.text((W / 2, top + k * lh), ln, font=sub_font, fill=(255, 245, 210), anchor="mt")
            ff.stdin.write(frame.tobytes())
        ff.stdin.close(); ff.wait()
    print("saved", out)


if __name__ == "__main__":
    main()
