GHMC automation PoC

Overview
- The web app now exposes a helper endpoint `/prepare_ghmc` which saves the current report as `ghmc_payload.json` in the project root when you click the "Lodge on GHMC (1‑click)" button on the Official Report page.
- The `scripts/ghmc_submit.py` script reads `ghmc_payload.json`, launches a visible browser (Playwright), opens the GHMC OTP page and optionally sends OTP (if you pass `--phone`), then pauses for you to complete OTP manually.
- After login the script navigates to the GHMC complaint entry page and attempts to autofill address, details and attach images. You must verify and submit the final form in the browser manually.

Quick start
1. Install dependencies (virtualenv recommended):
   pip install -r requirements.txt
   playwright install

2. Run your app (if not already running):
   python app.py

3. In the app: generate a report and click **Lodge on GHMC (1‑click)**. That will save `ghmc_payload.json` in the project root.

4. Run the script locally (headed browser):
   python scripts/ghmc_submit.py --payload ghmc_payload.json --phone 98XXXXXXXX

5. Complete the OTP in the browser and press ENTER in the terminal when logged in. Verify the pre-filled form and click Submit.

Notes & limitations
- The script is a PoC and tries multiple selectors but may need selector tweaks if GHMC changes their form. If some fields are not filled you can quickly paste the copied message into the complaint details field.
- OTP/CAPTCHA must be completed manually. Ensure automation is allowed by GHMC terms before heavy use.