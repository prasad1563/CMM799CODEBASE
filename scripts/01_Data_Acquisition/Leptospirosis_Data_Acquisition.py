import requests
import os

def download_wer_pdfs(volume: int, year: int, max_weeks: int = 52, out_dir: str = "wer_pdfs"):

    os.makedirs(out_dir, exist_ok=True)
    base = "https://www.epid.gov.lk/storage/post/pdfs"

    # All known filename patterns found on the site
    patterns = [
        "vol_{v}_no_{w}-english.pdf",
        "vol_{v}_no_{w}_english.pdf",
        "Vol_{v}_No_{w}-English.pdf",
        "Vol_{v}_No_{w}_English.pdf",
        "Vol_{v}_no_{w}-English.pdf",
        "vol_{v}_No_{w}-english.pdf",
        "vol_{v}_no_{w}-english_1.pdf",
        "vol_{v}_no_{w}-english_2.pdf",
        "vol_{v}_no_{w}-english_3.pdf",
    ]

    for week in range(1, max_weeks + 1):
        week_str = f"{week:02d}"
        downloaded = False

        for pattern in patterns:
            filename = pattern.format(v=volume, w=week_str)
            url = f"{base}/{filename}"

            try:
                resp = requests.get(url, timeout=10)
            except Exception as e:
                print(f"[ERROR] {url} -> {e}")
                continue

            if resp.status_code == 200 and "pdf" in resp.headers.get('Content-Type', '').lower():
                out_file = os.path.join(out_dir, f"Vol_{volume}_No_{week_str}.pdf")
                with open(out_file, "wb") as f:
                    f.write(resp.content)

                print(f"[OK] Downloaded week {week_str} using pattern: {filename}")
                downloaded = True
                break

        if not downloaded:
            print(f"[MISSING] Week {week_str} not found in any known format")

if __name__ == "__main__":
    download_wer_pdfs(volume=50, year=2023, max_weeks=53, out_dir="WER_2023")
