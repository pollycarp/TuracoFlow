"""
Generate mock hospital receipt images for TuracoFlow testing.

Receipt 1 (receipt_approved.png):   Valid Kenya claim — should be APPROVED
Receipt 2 (receipt_over_limit.png): Claim amount exceeds policy max — should be REJECTED
Receipt 3 (receipt_low_confidence.png): Blurry/incomplete receipt — should trigger REVIEW
"""

from PIL import Image, ImageDraw, ImageFont, ImageFilter
import os

OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))

# Try to use a basic system font; fall back to default if not found
def get_font(size):
    font_paths = [
        "C:/Windows/Fonts/arial.ttf",
        "C:/Windows/Fonts/calibri.ttf",
        "C:/Windows/Fonts/consola.ttf",
    ]
    for path in font_paths:
        if os.path.exists(path):
            return ImageFont.truetype(path, size)
    return ImageFont.load_default()


def draw_receipt(draw, lines, font_small, font_medium, font_large, start_y=30):
    """Draw lines of text onto a receipt image."""
    y = start_y
    for line in lines:
        text, size = line
        font = font_large if size == "large" else font_medium if size == "medium" else font_small
        draw.text((40, y), text, fill="black", font=font)
        y += size_to_spacing(size)
    return y


def size_to_spacing(size):
    return {"large": 40, "medium": 28, "small": 22}.get(size, 22)


# ──────────────────────────────────────────────
# Receipt 1: APPROVED — valid Kenya HospiCash claim
# ──────────────────────────────────────────────
def create_receipt_approved():
    img = Image.new("RGB", (600, 700), color="white")
    draw = ImageDraw.Draw(img)

    font_large  = get_font(22)
    font_medium = get_font(16)
    font_small  = get_font(13)

    lines = [
        ("KENYATTA NATIONAL HOSPITAL", "large"),
        ("Hospital Road, Nairobi, Kenya", "small"),
        ("Tel: +254 20 272 6300", "small"),
        ("", "small"),
        ("─" * 55, "small"),
        ("INPATIENT DISCHARGE RECEIPT", "medium"),
        ("─" * 55, "small"),
        ("", "small"),
        ("Receipt No:    KNH-2026-03-4471", "small"),
        ("Date Issued:   28 March 2026", "small"),
        ("", "small"),
        ("PATIENT DETAILS", "medium"),
        ("Patient Name:  John Kamau Mwangi", "small"),
        ("ID Number:     34521876", "small"),
        ("Policy No:     TUR-KE-HC-001-88234", "small"),
        ("Date of Birth: 14 July 1990", "small"),
        ("", "small"),
        ("ADMISSION DETAILS", "medium"),
        ("Admitted:      24 March 2026", "small"),
        ("Discharged:    27 March 2026", "small"),
        ("Nights:        3", "small"),
        ("Ward:          General Medical Ward B", "small"),
        ("Attending:     Dr. Alice Njoroge", "small"),
        ("", "small"),
        ("DIAGNOSIS", "medium"),
        ("Primary:       Malaria (ICD-10: B54)", "small"),
        ("Secondary:     Mild Dehydration", "small"),
        ("", "small"),
        ("BILLING SUMMARY", "medium"),
        ("Consultation:      KES   500", "small"),
        ("Inpatient (3 nts): KES 3,600", "small"),
        ("Medication:        KES 1,200", "small"),
        ("Lab Tests:         KES   700", "small"),
        ("─" * 55, "small"),
        ("TOTAL CHARGED:     KES 6,000", "medium"),
        ("─" * 55, "small"),
        ("", "small"),
        ("Hospital Stamp: [KENYATTA NATIONAL HOSPITAL]", "small"),
        ("Authorized By:  Dr. Alice Njoroge", "small"),
    ]

    draw_receipt(draw, lines, font_small, font_medium, font_large)

    # Add a faint border
    draw.rectangle([10, 10, 589, 689], outline="gray", width=2)

    path = os.path.join(OUTPUT_DIR, "receipt_approved.png")
    img.save(path)
    print(f"Created: {path}")


# ──────────────────────────────────────────────
# Receipt 2: REJECTED — amount exceeds policy limit
# ──────────────────────────────────────────────
def create_receipt_over_limit():
    img = Image.new("RGB", (600, 700), color="white")
    draw = ImageDraw.Draw(img)

    font_large  = get_font(22)
    font_medium = get_font(16)
    font_small  = get_font(13)

    lines = [
        ("NAIROBI HOSPITAL", "large"),
        ("Argwings Kodhek Road, Nairobi, Kenya", "small"),
        ("Tel: +254 20 284 5000", "small"),
        ("", "small"),
        ("─" * 55, "small"),
        ("INPATIENT DISCHARGE RECEIPT", "medium"),
        ("─" * 55, "small"),
        ("", "small"),
        ("Receipt No:    NH-2026-03-7821", "small"),
        ("Date Issued:   29 March 2026", "small"),
        ("", "small"),
        ("PATIENT DETAILS", "medium"),
        ("Patient Name:  Grace Achieng Otieno", "small"),
        ("ID Number:     28871045", "small"),
        ("Policy No:     TUR-KE-HC-001-71092", "small"),
        ("Date of Birth: 02 February 1985", "small"),
        ("", "small"),
        ("ADMISSION DETAILS", "medium"),
        ("Admitted:      10 March 2026", "small"),
        ("Discharged:    25 March 2026", "small"),
        ("Nights:        15", "small"),
        ("Ward:          Private Ward 3", "small"),
        ("Attending:     Dr. Samuel Kariuki", "small"),
        ("", "small"),
        ("DIAGNOSIS", "medium"),
        ("Primary:       Pneumonia (ICD-10: J18)", "small"),
        ("Secondary:     Pleural Effusion", "small"),
        ("", "small"),
        ("BILLING SUMMARY", "medium"),
        ("Consultation:      KES  2,000", "small"),
        ("Inpatient (15nts): KES 18,000", "small"),
        ("Medication:        KES  7,500", "small"),
        ("ICU (2 nights):    KES  8,000", "small"),
        ("Lab & Imaging:     KES  4,500", "small"),
        ("─" * 55, "small"),
        ("TOTAL CHARGED:     KES 40,000", "medium"),
        ("─" * 55, "small"),
        ("", "small"),
        ("Hospital Stamp: [NAIROBI HOSPITAL]", "small"),
        ("Authorized By:  Dr. Samuel Kariuki", "small"),
    ]

    draw_receipt(draw, lines, font_small, font_medium, font_large)
    draw.rectangle([10, 10, 589, 689], outline="gray", width=2)

    path = os.path.join(OUTPUT_DIR, "receipt_over_limit.png")
    img.save(path)
    print(f"Created: {path}")


# ──────────────────────────────────────────────
# Receipt 3: REVIEW — blurry, missing fields → low confidence
# ──────────────────────────────────────────────
def create_receipt_low_confidence():
    img = Image.new("RGB", (600, 700), color="white")
    draw = ImageDraw.Draw(img)

    font_large  = get_font(22)
    font_medium = get_font(16)
    font_small  = get_font(13)

    # Intentionally incomplete receipt
    lines = [
        ("SABURI MEDICAL CENTRE", "large"),
        ("Westlands, Nairobi", "small"),
        ("", "small"),
        ("─" * 55, "small"),
        ("RECEIPT", "medium"),
        ("─" * 55, "small"),
        ("", "small"),
        ("Date: 26/03/2026", "small"),
        ("", "small"),
        ("Patient: _______________", "small"),  # Missing name
        ("Policy No: ____________", "small"),   # Missing policy number
        ("", "small"),
        ("Admitted:  25 March 2026", "small"),
        ("Discharged: ___________", "small"),   # Missing discharge date
        ("Nights: ?", "small"),                 # Unclear nights
        ("", "small"),
        ("Diagnosis: Fever / Possible infection", "small"),  # Vague — no ICD-10
        ("", "small"),
        ("Treatment: Medication + Observation", "small"),
        ("", "small"),
        ("Amount: KES 3,500", "small"),
        ("", "small"),
        ("Stamp: [ILLEGIBLE]", "small"),
    ]

    draw_receipt(draw, lines, font_small, font_medium, font_large)
    draw.rectangle([10, 10, 589, 689], outline="gray", width=2)

    # Apply blur to simulate a bad photo taken on WhatsApp
    img = img.filter(ImageFilter.GaussianBlur(radius=1.8))

    # Add noise-like texture
    import random
    pixels = img.load()
    for _ in range(3000):
        x = random.randint(0, img.width - 1)
        y = random.randint(0, img.height - 1)
        noise = random.randint(-30, 30)
        r, g, b = pixels[x, y]
        pixels[x, y] = (
            max(0, min(255, r + noise)),
            max(0, min(255, g + noise)),
            max(0, min(255, b + noise)),
        )

    path = os.path.join(OUTPUT_DIR, "receipt_low_confidence.png")
    img.save(path)
    print(f"Created: {path}")


if __name__ == "__main__":
    create_receipt_approved()
    create_receipt_over_limit()
    create_receipt_low_confidence()
    print("\nAll 3 receipt images generated successfully.")
