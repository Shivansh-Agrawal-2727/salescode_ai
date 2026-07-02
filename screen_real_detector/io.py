from pathlib import Path

from PIL import Image, ImageOps

SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp", ".heic", ".heif"}


def register_heif_support() -> bool:
    try:
        from pillow_heif import register_heif_opener
    except ImportError:
        return False

    register_heif_opener()
    return True


def iter_images(folder: Path):
    for path in sorted(folder.rglob("*")):
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS:
            yield path


def open_image(path: Path, max_size: int = 512) -> Image.Image:
    has_heif = register_heif_support()
    if path.suffix.lower() in {".heic", ".heif"} and not has_heif:
        raise RuntimeError("Install pillow-heif to read HEIC/HEIF images: pip install pillow-heif")

    image = Image.open(path)
    image = ImageOps.exif_transpose(image).convert("RGB")
    
    w, h = image.size
    left = max(0, (w - max_size) // 2)
    top = max(0, (h - max_size) // 2)
    right = min(w, left + max_size)
    bottom = min(h, top + max_size)
    
    image = image.crop((left, top, right, bottom))
    if image.size != (max_size, max_size):
        image = image.resize((max_size, max_size), Image.Resampling.LANCZOS)
    return image

