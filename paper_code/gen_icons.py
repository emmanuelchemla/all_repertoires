"""Save emoji as PNG icons for fig_landmark_v2.py (macOS — uses Swift/NSImage)."""
import subprocess
import tempfile
import os
from pathlib import Path

ICON_DIR = Path(__file__).parent / "icons"
ICON_DIR.mkdir(exist_ok=True)

ICONS = {
    "sem_bubble":   "💬",
    "acc_speaker":  "🔊",
    "cls_aves":     "🐦",
    "cls_mammalia": "🐾",
    "cls_amphibia": "🐸",
}

SWIFT = """
import AppKit

let emojis: [(String, String)] = [
    {pairs}
]

let size: CGFloat = 160
let iconDir = "{icon_dir}"

for (emoji, name) in emojis {{
    let img = NSImage(size: NSSize(width: size, height: size))
    img.lockFocus()
    let attrs: [NSAttributedString.Key: Any] = [
        .font: NSFont.systemFont(ofSize: size * 0.85)
    ]
    NSAttributedString(string: emoji, attributes: attrs).draw(at: NSPoint(x: 0, y: 0))
    img.unlockFocus()
    guard let tiff = img.tiffRepresentation,
          let rep = NSBitmapImageRep(data: tiff),
          let png = rep.representation(using: .png, properties: [:]) else {{ continue }}
    try! png.write(to: URL(fileURLWithPath: iconDir + "/" + name + ".png"))
    print("  " + name + ".png")
}}
"""


if __name__ == "__main__":
    pairs = ",\n    ".join(f'("{v}", "{k}")' for k, v in ICONS.items())
    swift_code = SWIFT.format(pairs=pairs, icon_dir=str(ICON_DIR))

    with tempfile.NamedTemporaryFile(suffix=".swift", mode="w", delete=False) as f:
        f.write(swift_code)
        swift_file = f.name

    print("Generating icons …")
    result = subprocess.run(["swift", swift_file], capture_output=True, text=True)
    print(result.stdout, end="")
    if result.returncode != 0:
        print("ERROR:", result.stderr[:400])
    os.unlink(swift_file)
    print("Done.")
