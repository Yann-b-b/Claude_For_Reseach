#!/usr/bin/env python3
import re
import time
import csv
import sys
from typing import List, Dict, Optional

import requests
from requests.exceptions import SSLError
from bs4 import BeautifulSoup
import certifi

BASE = "https://aps.unmc.edu"
LIST_URL = BASE + "/database/anti"


def fetch(url: str, session: Optional[requests.Session] = None) -> str:
    s = session or requests.Session()
    try:
        resp = s.get(url, timeout=30)
        resp.raise_for_status()
        return resp.text
    except SSLError:
        # Fallback: try http if https fails due to cert issues
        if url.startswith("https://"):
            alt = url.replace("https://", "http://", 1)
            resp = s.get(alt, timeout=30, allow_redirects=True)
            resp.raise_for_status()
            return resp.text
        raise


def make_session(insecure: bool = False) -> requests.Session:
    s = requests.Session()
    s.headers.update({
        "User-Agent": "Mozilla/5.0 (compatible; APD-Scraper/1.0)",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    })
    # Control TLS verification
    if insecure:
        s.verify = False
    else:
        s.verify = certifi.where()
    return s


def parse_list(html: str) -> List[str]:
    soup = BeautifulSoup(html, "lxml")
    links = []
    for a in soup.select('a[href^="/database/peptide/"]'):
        href = a.get("href")
        if href and re.match(r"^/database/peptide/\w+$", href):
            links.append(BASE + href)
    # Deduplicate while preserving order
    seen = set()
    out = []
    for u in links:
        if u not in seen:
            seen.add(u)
            out.append(u)
    return out


def extract_text(soup: BeautifulSoup, label: str) -> Optional[str]:
    # Look for a label like "APD ID:", "Name/Class:", "Sequence:" etc.
    patt = re.compile(rf"^{re.escape(label)}\s*", re.I)
    for strong in soup.find_all(['b','strong']):
        if strong.get_text(strip=True) and patt.match(strong.get_text(strip=True)):
            # Next sibling or parent text
            parent = strong.parent
            text = parent.get_text(" ", strip=True)
            text = re.sub(r"^" + re.escape(label) + r"\s*", "", text, flags=re.I)
            return text
    return None


def parse_detail(html: str) -> Dict[str, str]:
    soup = BeautifulSoup(html, "lxml")

    # Name: from "Name/Class:" field; fallback to page title
    name = extract_text(soup, "Name/Class:")
    if not name:
        h1 = soup.find(['h1','h2'])
        name = h1.get_text(strip=True) if h1 else None

    # Sequence: lines often labeled "Sequence:" followed by uppercase letters
    seq_block = extract_text(soup, "Sequence:")
    sequence = None
    if seq_block:
        # keep only letters
        seq = re.sub(r"[^A-Za-z]", "", seq_block).upper()
        if seq:
            sequence = seq

    # Bacterial targets: from Activity sections mentioning species; also specific lines like "Active against ..."
    targets: List[str] = []
    # 1) Look for a line starting with "Activity:" then parse species tokens separated by commas and semicolons
    act = None
    # First try the shorter label
    act = extract_text(soup, "Activity:") or extract_text(soup, "Activity")
    if act:
        # Extract canonical species names like E. coli, P. aeruginosa, S. aureus, etc.
        # Also longer names (two-token binomials)
        species_patterns = [
            r"E\.\s*coli",
            r"S\.\s*aureus",
            r"P\.\s*aeruginosa",
            r"B\.\s*subtilis",
            r"S\.\s*pneumoniae",
            r"[A-Z][a-z]+\s+[a-z]+"  # generic Genus species
        ]
        for pat in species_patterns:
            for m in re.finditer(pat, act):
                targets.append(m.group(0))

    # 2) Parse "Active against ..." paragraph if present
    text = soup.get_text("\n", strip=True)
    m = re.search(r"Active against\s*:(.*)", text, re.I)
    if m:
        seg = m.group(1)
        # split by commas and parse species tokens again
        for token in re.split(r"[,;]", seg):
            token = token.strip()
            if re.match(r"^[A-Z][a-z]+\s+[a-z]+$", token) or re.match(r"^[A-Z]\.\s*[a-z]+$", token):
                targets.append(token)

    # Deduplicate/normalize targets
    norm_targets = []
    seen = set()
    for t in targets:
        t = re.sub(r"\s+", " ", t).strip()
        if t and t not in seen:
            seen.add(t)
            norm_targets.append(t)

    return {
        "name": name or "",
        "sequence": sequence or "",
        "targets": "; ".join(norm_targets)
    }


def scrape(limit: int = 100, delay: float = 0.5, outfile: str = "apd_scrape.csv", insecure: bool = False):
    session = make_session(insecure=insecure)
    print("Fetching list page...", file=sys.stderr)
    html = fetch(LIST_URL, session)
    detail_urls = parse_list(html)
    if limit:
        detail_urls = detail_urls[:limit]
    print(f"Found {len(detail_urls)} entries", file=sys.stderr)

    rows: List[Dict[str, str]] = []
    for i, url in enumerate(detail_urls, 1):
        try:
            page = fetch(url, session)
            data = parse_detail(page)
            data["url"] = url
            rows.append(data)
            print(f"[{i}/{len(detail_urls)}] {data.get('name','')} - {len(data.get('sequence',''))} aa", file=sys.stderr)
            time.sleep(delay)
        except Exception as e:
            print(f"Error on {url}: {e}", file=sys.stderr)

    with open(outfile, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["name","sequence","targets","url"])
        w.writeheader()
        w.writerows(rows)
    print(f"Saved {len(rows)} rows to {outfile}")


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description="Scrape APD peptide name/sequence/targets")
    p.add_argument("--limit", type=int, default=100)
    p.add_argument("--delay", type=float, default=0.5)
    p.add_argument("--out", type=str, default="apd_scrape.csv")
    p.add_argument("--insecure", action="store_true", help="Disable TLS verification and allow http fallback")
    args = p.parse_args()
    scrape(limit=args.limit, delay=args.delay, outfile=args.out, insecure=bool(args.insecure))


