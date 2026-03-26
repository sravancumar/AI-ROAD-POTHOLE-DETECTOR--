#!/usr/bin/env python3
"""GHMC complaint autofill PoC using Playwright (headed browser).

Usage:
  python scripts/ghmc_submit.py [--payload PATH] [--phone PHONE]

Notes:
 - This script opens the GHMC grievance page in a visible browser.
 - If you provide --phone the script will attempt to send OTP automatically; otherwise you can enter the phone in the page.
 - Manual OTP entry is required. After you complete OTP in the browser, press ENTER in the terminal to continue the flow.
 - The script reads 'ghmc_payload.json' created by the web app (route '/prepare_ghmc').
"""
import argparse
import json
import os
import sys
from pathlib import Path
from time import sleep

try:
    from playwright.sync_api import sync_playwright
except Exception as e:
    print("Playwright is not installed or needs install. Run: pip install -r requirements.txt && playwright install")
    raise

BASE_DIR = Path(__file__).resolve().parents[1]
DEFAULT_PAYLOAD = BASE_DIR / 'ghmc_payload.json'

SEL_TRIES = 3

def safe_fill(page, selectors, value):
    for sel in selectors:
        try:
            if page.query_selector(sel):
                page.fill(sel, value)
                print(f"Filled {sel} -> {value}")
                return True
        except Exception:
            pass
    return False


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--payload', default=str(DEFAULT_PAYLOAD), help='Path to ghmc_payload.json')
    p.add_argument('--phone', default=None, help='Phone number to pre-fill for OTP')
    args = p.parse_args()

    payload_path = Path(args.payload)
    if not payload_path.exists():
        print(f"Payload file not found: {payload_path}\nRun the web app and click 'Lodge on GHMC' to create it.")
        sys.exit(1)

    payload = json.loads(payload_path.read_text())
    print("Loaded payload:")
    print(json.dumps(payload, indent=2))

    images = payload.get('images', [])
    for img in images:
        if not Path(img).exists():
            print(f"Warning: image not found: {img}")

    with sync_playwright() as pw:
        browser = pw.chromium.launch(headless=False)
        context = browser.new_context(viewport={'width':1200,'height':900})
        page = context.new_page()

        # Open GHMC OTP page
        page.goto('https://igs.ghmc.gov.in/operator/send_otp_mobile', timeout=60000)
        print("Opened GHMC OTP page in the browser.")

        # If phone provided, try to fill and send OTP
        if args.phone:
            try:
                page.fill('#mbno', args.phone)
                # Click 'Send OTP' button (try a few variants)
                for t in ['text=Send OTP', 'text=Send', 'button:has-text("Send OTP")', 'button:has-text("Send")']:
                    try:
                        btn = page.query_selector(t)
                        if btn:
                            btn.click()
                            print('Sent OTP (button clicked).')
                            break
                    except Exception:
                        pass
            except Exception as e:
                print('Could not auto-fill phone number:', e)

        print('\nManual step: Complete OTP in the opened browser and ensure you are logged in.\nAfter login (OTP), press ENTER here to continue the automation to the complaint form.')
        input('Press ENTER after login...')

        # Navigate to complaint entry page
        page.goto('https://igs.ghmc.gov.in/Grievance/GrievanceEntry', timeout=60000)
        sleep(1)

        # Attempt to fill address/location inputs
        address = payload.get('address', '')
        lat = str(payload.get('lat', ''))
        lon = str(payload.get('lon', ''))
        message = payload.get('message', '')

        print('\nFilling Address / Location fields...')
        addr_selectors = ['input[name*=Address i]', 'input[name*=address i]', 'input[id*=address i]', 'input[placeholder*=Address i]', 'input[name*=location i]', 'input[id*=location i]', 'input[placeholder*=Location i]']
        # fallback more generic selectors
        addr_selectors_generic = ['input[type="text"]', 'textarea']

        filled = safe_fill(page, addr_selectors, address)
        if not filled:
            # try text inputs and set first available as address
            for sel in addr_selectors_generic:
                try:
                    el = page.query_selector(sel)
                    if el:
                        el.fill(address)
                        print(f"Filled generic {sel} as address")
                        break
                except Exception:
                    pass

        # Fill lat / lon if input fields present
        try:
            page.evaluate("(lat, lon) => { const la = document.querySelector('input[name*=lat], input[id*=lat]'); const lo = document.querySelector('input[name*=lon], input[id*=lon]'); if(la) la.value = lat; if(lo) lo.value = lon; }", lat, lon)
            print('Set Latitude/Longitude fields if present.')
        except Exception:
            pass

        # Fill description / details
        print('\nFilling description...')
        desc_selectors = ['textarea[name*=Grievance i]', 'textarea[name*=Description i]', 'textarea[name*=details i]', 'textarea', 'input[name*=description i]']
        if not safe_fill(page, desc_selectors, message):
            print('Warning: could not find description textarea. Please paste manually if needed.')

        # Attach images if file input exists
        print('\nAttempting to attach images...')
        try:
            file_input = None
            for sel in ['input[type=file]', 'input[type="file"]']: file_input = page.query_selector(sel) if not file_input else file_input
            if file_input:
                abs_imgs = [str(Path(p).resolve()) for p in images if Path(p).exists()]
                if abs_imgs:
                    file_input.set_input_files(abs_imgs)
                    print(f'Attached {len(abs_imgs)} images.')
                else:
                    print('No valid images to attach (files missing).')
            else:
                print('No file input found on the form; please attach images manually.')
        except Exception as e:
            print('Attach images failed:', e)

        print('\nDone: The GHMC grievance form should be pre-filled in the browser. Please verify the fields and click Submit manually.\nThe script will keep the browser open so you can finish the flow.')
        try:
            input('Press ENTER to close the browser and exit when you are finished (browser will remain open until you close it).')
        except KeyboardInterrupt:
            pass
        try:
            context.close()
            browser.close()
        except Exception:
            pass

if __name__ == '__main__':
    main()
