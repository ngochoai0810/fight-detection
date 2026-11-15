"""Bundle Google Fonts, including Vietnamese glyphs, for offline dashboard use."""
import re
import urllib.request
from pathlib import Path


def main():
    target = Path(__file__).resolve().parents[1] / "src" / "assets" / "fonts"
    target.mkdir(parents=True, exist_ok=True)
    url = "https://fonts.googleapis.com/css2?family=Baloo+2:wght@700;800&family=Nunito:wght@600;700;800;900&display=swap"
    request = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    css = urllib.request.urlopen(request, timeout=30).read().decode()
    for index, source in enumerate(dict.fromkeys(re.findall(r"url\((https://[^)]+)\)", css))):
        suffix = ".woff2" if ".woff2" in source else ".ttf"
        filename = f"dashboard-{index}{suffix}"
        data = urllib.request.urlopen(source, timeout=30).read()
        (target / filename).write_bytes(data)
        css = css.replace(source, f"/assets/fonts/{filename}")
    (target / "fonts.css").write_text(css, encoding="utf-8")
    for family in ("baloo2", "nunito"):
        license_url = f"https://raw.githubusercontent.com/google/fonts/main/ofl/{family}/OFL.txt"
        (target / f"{family}-OFL.txt").write_bytes(urllib.request.urlopen(license_url, timeout=30).read())
    print(f"Downloaded fonts to {target}")


if __name__ == "__main__":
    main()
