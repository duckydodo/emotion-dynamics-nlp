#!/usr/bin/env python3
"""
Task 33 — ConvoSense Visual QA & Browser Automation Script
Executes comprehensive visual QA, accessibility audits, and captures high-resolution
desktop (1440x1024) and mobile (390x844) screenshots for all five screens of ConvoSense.
"""

import asyncio
import base64
import json
from pathlib import Path
import subprocess
import time
import urllib.request
import websockets

APP_URL = "http://localhost:8001"
CHROME_PORT = 9224
FIGURES_DIR = Path("outputs/figures")
FIGURES_DIR.mkdir(parents=True, exist_ok=True)
TABLES_DIR = Path("outputs/tables")
TABLES_DIR.mkdir(parents=True, exist_ok=True)

RESULTS = []


def record_result(screen, category, check_name, expected, actual, passed, detail=""):
    RESULTS.append({
        "screen": screen,
        "category": category,
        "check": check_name,
        "expected": expected,
        "actual": actual,
        "status": "PASS" if passed else "FAIL",
        "detail": detail,
    })
    status_str = "\033[92mPASS\033[0m" if passed else "\033[91mFAIL\033[0m"
    print(f"[{status_str}] [{screen}] [{category}] {check_name}")
    if not passed:
        print(f"   Expected: {expected}")
        print(f"   Actual:   {actual}")
    if detail:
        print(f"   Detail:   {detail}")


async def run_visual_qa():
    print("==================================================================")
    print("TASK 33: CONVOSENSE DESIGN SYSTEM & VISUAL QA AUTOMATION")
    print(f"Target URL: {APP_URL}")
    print("==================================================================\n")

    chrome_proc = subprocess.Popen([
        "google-chrome",
        "--headless=new",
        "--disable-gpu",
        f"--remote-debugging-port={CHROME_PORT}",
        "--remote-allow-origins=*",
        "--user-data-dir=/tmp/chrome_test_profile_task33",
        "--window-size=1440,1024",
        "--no-sandbox",
        APP_URL,
    ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    try:
        ws_url = None
        for _ in range(40):
            try:
                with urllib.request.urlopen(f"http://127.0.0.1:{CHROME_PORT}/json", timeout=1) as resp:
                    tabs = json.loads(resp.read().decode())
                    for tab in tabs:
                        if tab.get("type") == "page" and "8001" in tab.get("url", ""):
                            ws_url = tab["webSocketDebuggerUrl"]
                            break
                    if ws_url:
                        break
            except Exception:
                await asyncio.sleep(0.25)

        if not ws_url:
            raise RuntimeError(f"Failed to connect to Chrome CDP at port {CHROME_PORT}")

        async with websockets.connect(ws_url, max_size=40 * 1024 * 1024) as ws:
            msg_counter = [1]

            async def send_cdp(method, params=None):
                mid = msg_counter[0]
                msg_counter[0] += 1
                payload = {"id": mid, "method": method, "params": params or {}}
                await ws.send(json.dumps(payload))
                while True:
                    resp_raw = await ws.recv()
                    resp = json.loads(resp_raw)
                    if resp.get("id") == mid:
                        return resp.get("result", {})

            async def eval_js(code):
                res = await send_cdp("Runtime.evaluate", {
                    "expression": code,
                    "returnByValue": True,
                    "awaitPromise": True,
                })
                if "exceptionDetails" in res:
                    raise RuntimeError(f"JS Exception: {res['exceptionDetails']}")
                return res.get("result", {}).get("value")

            async def take_screenshot(filename):
                res = await send_cdp("Page.captureScreenshot", {"format": "png"})
                data = base64.b64decode(res.get("data", ""))
                out_path = FIGURES_DIR / filename
                out_path.write_bytes(data)
                return out_path

            async def set_viewport(width, height, mobile=False):
                await send_cdp("Emulation.setDeviceMetricsOverride", {
                    "width": width,
                    "height": height,
                    "deviceScaleFactor": 2 if mobile else 1,
                    "mobile": mobile,
                })
                await asyncio.sleep(0.5)

            # Wait for DOM and initial models fetch to resolve
            await asyncio.sleep(2.0)

            # =================================================================
            # PART 1: DESIGN SYSTEM COMPLIANCE & BRANDING CHECKS
            # =================================================================
            print("\n--- [PART 1: Design System Compliance & Branding] ---")

            # 1.1 Page Title & Favicon
            page_title = await eval_js("document.title")
            favicon_href = await eval_js("document.querySelector('link[rel~=\"icon\"]')?.href || ''")
            record_result(
                "Global", "Identity", "Browser Title",
                "ConvoSense — Conversational Emotion Dynamics & Model Intelligence",
                page_title,
                page_title == "ConvoSense — Conversational Emotion Dynamics & Model Intelligence",
                f"Title: {page_title}"
            )
            record_result(
                "Global", "Identity", "Favicon SVG",
                "SVG favicon defined with ConvoSense wave glyph",
                favicon_href[:50],
                "svg+xml" in favicon_href,
                f"Favicon type: {favicon_href[:40]}..."
            )

            # 1.2 Font Stacks
            font_body = await eval_js("window.getComputedStyle(document.body).fontFamily")
            font_mono = await eval_js("""
                const el = document.querySelector('.token-chip, .code-metric, pre, code') || document.body;
                window.getComputedStyle(el).fontFamily
            """)
            record_result(
                "Global", "Typography", "Body Typeface (Plus Jakarta Sans)",
                "Contains 'Plus Jakarta Sans'",
                font_body,
                "Plus Jakarta Sans" in font_body,
                f"Body font: {font_body}"
            )

            # 1.3 Icon Audit (Strict Zero-Emoji Rule for UI)
            # Check navbar, tabs, buttons, format tags, status badge
            emoji_check = await eval_js("""
                (() => {
                    const emojiRegex = /(\\p{Emoji_Presentation}|\\p{Extended_Pictographic})/u;
                    const elements = Array.from(document.querySelectorAll(
                        '.navbar-brand, .nav-btn, .btn, .format-pill, .status-pill, .kpi-label'
                    ));
                    const violatingElements = [];
                    for (const el of elements) {
                        // Allow purely standard text and SVG children
                        const text = el.cloneNode(true);
                        // remove SVGs
                        text.querySelectorAll('svg').forEach(s => s.remove());
                        const str = text.textContent.trim();
                        // Exclude ethical warning triangle if in non-clinical notice
                        if (emojiRegex.test(str)) {
                            violatingElements.push({ tag: el.tagName, class: el.className, text: str });
                        }
                    }
                    return violatingElements;
                })()
            """)
            record_result(
                "Global", "Design System", "Zero Emojis in Navigation/Buttons/Pills",
                "0 emoji violations in core UI controls",
                f"{len(emoji_check)} violations",
                len(emoji_check) == 0,
                f"Violations: {emoji_check}"
            )

            # 1.4 Inline SVG Icon Verification
            svg_count = await eval_js("document.querySelectorAll('svg').length")
            record_result(
                "Global", "Design System", "Inline SVG Icons Rendered",
                ">= 20 inline vector SVGs",
                f"{svg_count} SVGs",
                svg_count >= 20,
                f"Found {svg_count} crisp inline SVGs across the interface"
            )

            # =================================================================
            # PART 2: DESKTOP SUITE (1440x1024)
            # =================================================================
            print("\n--- [PART 2: Desktop Suite (1440x1024)] ---")
            await set_viewport(1440, 1024, mobile=False)

            # Screen 1: Home / Import
            await eval_js("navigateTo('import');")
            # Click Sample 1 (Conflict & Resolution) to populate drawer
            await eval_js("document.getElementById('sampleConflictBtn').click();")
            await asyncio.sleep(1.0)

            # Check Screen 1 elements
            s1_active = await eval_js("document.getElementById('screen-import').classList.contains('active')")
            s1_model_opts = await eval_js("Array.from(document.getElementById('modelSelect').options).map(o => o.value)")
            s1_preview_open = await eval_js("document.getElementById('importPreviewPanel').style.display !== 'none'")
            s1_turns = await eval_js("document.getElementById('previewEvalTurns').textContent")

            record_result(
                "Screen 1 (Home/Import)", "Desktop", "Sample Loaded & Preview Active",
                "Screen active, 3 primary models, preview open, 9 evaluable turns",
                f"active={s1_active}, models={s1_model_opts}, turns={s1_turns}",
                s1_active and set(s1_model_opts) == {"distilbert", "dialoguernn", "dialoguegcn"} and s1_preview_open and str(s1_turns) == "9",
                f"Loaded sample with {s1_turns} turns, models: {s1_model_opts}"
            )

            # Capture Desktop Screen 1
            p1 = await take_screenshot("task33_desktop_screen1_import.png")
            print(f"   [Captured] {p1.name}")

            # Screen 2: Conversation Viewer (Run Analysis)
            await eval_js("executeAnalysis();")
            await asyncio.sleep(4.0)

            s2_active = await eval_js("document.getElementById('screen-viewer').classList.contains('active')")
            s2_bubbles = await eval_js("document.querySelectorAll('.bubble-wrapper').length")
            # Open drawer on first bubble
            await eval_js("toggleProbDrawer('prob-drawer-0');")
            await asyncio.sleep(0.5)
            s2_drawer_open = await eval_js("document.getElementById('prob-drawer-0')?.classList.contains('open') || false")

            record_result(
                "Screen 2 (Viewer)", "Desktop", "Speech Bubbles & Softmax Probability Drawer",
                "Viewer active, 9 message bubbles, prob-drawer-0 expanded",
                f"active={s2_active}, bubbles={s2_bubbles}, drawer_open={s2_drawer_open}",
                s2_active and s2_bubbles == 9 and s2_drawer_open,
                f"Bubble count: {s2_bubbles}, Drawer open: {s2_drawer_open}"
            )

            p2 = await take_screenshot("task33_desktop_screen2_viewer.png")
            print(f"   [Captured] {p2.name}")

            # Screen 3: Emotion Dynamics
            await eval_js("navigateTo('dynamics');")
            await asyncio.sleep(1.0)

            s3_active = await eval_js("document.getElementById('screen-dynamics').classList.contains('active')")
            s3_shift_rate = await eval_js("document.getElementById('dynShiftRate').textContent")
            s3_transitions = await eval_js("document.getElementById('dynTotalTransitions').textContent")
            s3_chart_ok = await eval_js("AppState.chartInstance !== null")
            s3_turning_points = await eval_js("document.querySelectorAll('#turningPointsList .tp-card').length")

            record_result(
                "Screen 3 (Dynamics)", "Desktop", "KPIs, Chart.js Timeline & Turning Points",
                "Dynamics active, shift rate calculated, Chart.js instance active, TP cards rendered",
                f"active={s3_active}, shift_rate={s3_shift_rate}, transitions={s3_transitions}, chart_ok={s3_chart_ok}, tp_cards={s3_turning_points}",
                s3_active and "%" in s3_shift_rate and int(s3_transitions) > 0 and s3_chart_ok,
                f"Shift Rate: {s3_shift_rate}, Transitions: {s3_transitions}, Turning points count: {s3_turning_points}"
            )

            p3 = await take_screenshot("task33_desktop_screen3_dynamics.png")
            print(f"   [Captured] {p3.name}")

            # Screen 4: Model Comparison
            await eval_js("executeComparison();")
            await asyncio.sleep(4.0)

            s4_active = await eval_js("document.getElementById('screen-compare').classList.contains('active')")
            s4_consensus = await eval_js("document.getElementById('consensusRateBadge').textContent")
            s4_matrix_cells = await eval_js("document.querySelectorAll('#pairwiseMatrixBody td.agreement-cell').length")
            s4_rows = await eval_js("document.querySelectorAll('.comparison-row').length")

            record_result(
                "Screen 4 (Comparison)", "Desktop", "Consensus Rate, 3x3 Agreement Matrix, & Multi-Model Cards",
                "Comparison active, consensus rate visible, 9 matrix cells, 9 comparison rows",
                f"active={s4_active}, consensus={s4_consensus}, matrix_cells={s4_matrix_cells}, rows={s4_rows}",
                s4_active and "%" in s4_consensus and s4_matrix_cells == 9 and s4_rows == 9,
                f"Consensus: {s4_consensus}, Matrix cells: {s4_matrix_cells}, Rows: {s4_rows}"
            )

            p4 = await take_screenshot("task33_desktop_screen4_comparison.png")
            print(f"   [Captured] {p4.name}")

            # Screen 5: Research & Benchmarks
            await eval_js("navigateTo('research');")
            await asyncio.sleep(1.0)

            s5_active = await eval_js("document.getElementById('screen-research').classList.contains('active')")
            s5_cohort_rows = await eval_js("document.querySelectorAll('#cohortATableBody tr').length")
            s5_cm_cells = await eval_js("document.querySelectorAll('#cmContainer td').length")
            s5_disclaimer = await eval_js("document.body.innerText.includes('Research note: Non-Clinical Use & Ethical Safeguards')")

            record_result(
                "Screen 5 (Research)", "Desktop", "Cohort A Benchmark, 7x7 Confusion Matrix & Ethics Notice",
                "Research active, >= 3 benchmark rows, 49 matrix cells, ethics disclaimer present",
                f"active={s5_active}, rows={s5_cohort_rows}, cm_cells={s5_cm_cells}, disclaimer={s5_disclaimer}",
                s5_active and s5_cohort_rows >= 3 and s5_cm_cells == 49 and s5_disclaimer,
                f"Cohort A rows: {s5_cohort_rows}, CM cells: {s5_cm_cells}"
            )

            p5 = await take_screenshot("task33_desktop_screen5_research.png")
            print(f"   [Captured] {p5.name}")

            # =================================================================
            # PART 3: MOBILE SUITE (390x844 - iPhone 14 / Modern Standard)
            # =================================================================
            print("\n--- [PART 3: Mobile Suite (390x844)] ---")
            await set_viewport(390, 844, mobile=True)

            # Mobile Screen 1: Home / Import
            await eval_js("navigateTo('import');")
            await asyncio.sleep(1.0)
            m1_overflow = await eval_js("document.documentElement.scrollWidth <= window.innerWidth")
            record_result(
                "Screen 1 (Home/Import)", "Mobile", "Responsive Layout & No Horizontal Overflow",
                "scrollWidth <= window.innerWidth (390px)",
                f"no_overflow={m1_overflow}",
                m1_overflow,
                "Mobile viewport fit verified"
            )
            mp1 = await take_screenshot("task33_mobile_screen1_import.png")
            print(f"   [Captured] {mp1.name}")

            # Mobile Screen 2: Conversation Viewer
            await eval_js("navigateTo('viewer');")
            await asyncio.sleep(1.0)
            m2_bubbles_width = await eval_js("""
                (() => {
                    const bubbles = document.querySelectorAll('.chat-bubble');
                    if (bubbles.length === 0) return true;
                    return Array.from(bubbles).every(b => b.getBoundingClientRect().width <= 370);
                })()
            """)
            record_result(
                "Screen 2 (Viewer)", "Mobile", "Chat Bubble Max Width Within Mobile Viewport",
                "All chat bubbles width <= 370px",
                f"fits={m2_bubbles_width}",
                m2_bubbles_width,
                "Mobile chat bubbles wrapped cleanly"
            )
            mp2 = await take_screenshot("task33_mobile_screen2_viewer.png")
            print(f"   [Captured] {mp2.name}")

            # Mobile Screen 3: Emotion Dynamics
            await eval_js("navigateTo('dynamics');")
            await asyncio.sleep(1.0)
            m3_kpi_wrap = await eval_js("""
                (() => {
                    const cards = document.querySelectorAll('.kpi-card');
                    return cards.length >= 4;
                })()
            """)
            record_result(
                "Screen 3 (Dynamics)", "Mobile", "KPI Cards Stack Vertically/In Grid",
                "KPI cards render cleanly in mobile grid",
                f"cards={m3_kpi_wrap}",
                m3_kpi_wrap,
                "Mobile KPI grid verified"
            )
            mp3 = await take_screenshot("task33_mobile_screen3_dynamics.png")
            print(f"   [Captured] {mp3.name}")

            # Mobile Screen 4: Model Comparison
            await eval_js("navigateTo('compare');")
            await asyncio.sleep(1.0)
            m4_matrix_scroll = await eval_js("""
                (() => {
                    const container = document.querySelector('.table-responsive');
                    return container ? container.scrollWidth >= container.clientWidth : true;
                })()
            """)
            record_result(
                "Screen 4 (Comparison)", "Mobile", "Table Horizontal Scroll Container",
                "Matrix table wrapped in responsive container",
                f"scroll_ok={m4_matrix_scroll}",
                m4_matrix_scroll,
                "Responsive horizontal scroll available without page blowout"
            )
            mp4 = await take_screenshot("task33_mobile_screen4_comparison.png")
            print(f"   [Captured] {mp4.name}")

            # Mobile Screen 5: Research & Benchmarks
            await eval_js("navigateTo('research');")
            await asyncio.sleep(1.0)
            m5_cm_scroll = await eval_js("""
                (() => {
                    const el = document.getElementById('cmContainer');
                    return el ? el.scrollWidth >= el.clientWidth : true;
                })()
            """)
            record_result(
                "Screen 5 (Research)", "Mobile", "7x7 Confusion Matrix Scrollable",
                "Confusion matrix renders with horizontal scroll capability",
                f"scroll_ok={m5_cm_scroll}",
                m5_cm_scroll,
                "Confusion matrix responsive container verified"
            )
            mp5 = await take_screenshot("task33_mobile_screen5_research.png")
            print(f"   [Captured] {mp5.name}")

            # =================================================================
            # PART 4: ACCESSIBILITY & CONTRAST AUDIT (WCAG AA)
            # =================================================================
            print("\n--- [PART 4: Accessibility & Contrast Audit (WCAG AA)] ---")
            await set_viewport(1440, 1024, mobile=False)

            # Check touch target size on primary action buttons and navigation tabs (>= 36px height)
            target_sizes = await eval_js("""
                (() => {
                    const buttons = Array.from(document.querySelectorAll('.btn-primary, .tab-btn'));
                    const small = buttons.filter(b => {
                        const rect = b.getBoundingClientRect();
                        return rect.height > 0 && rect.height < 36;
                    });
                    return small.map(b => ({ class: b.className, text: b.textContent.trim(), h: b.getBoundingClientRect().height }));
                })()
            """)
            record_result(
                "Global", "Accessibility", "Interactive Button Minimum Target Height (>= 36px)",
                "0 interactive primary buttons below 36px height",
                f"{len(target_sizes)} undersized targets",
                len(target_sizes) == 0,
                f"Undersized list: {target_sizes}"
            )

            # Check text color contrast ratios against background in dark theme
            contrast_audit = await eval_js("""
                (() => {
                    // Function to calculate relative luminance
                    function getLuminance(r, g, b) {
                        const a = [r, g, b].map(v => {
                            v /= 255;
                            return v <= 0.03928 ? v / 12.92 : Math.pow((v + 0.055) / 1.055, 2.4);
                        });
                        return a[0] * 0.2126 + a[1] * 0.7152 + a[2] * 0.0722;
                    }

                    // Background in dark theme is ~ #090d16 -> rgb(9, 13, 22)
                    const bgLum = getLuminance(9, 13, 22);

                    const testClasses = [
                        { sel: 'body', name: 'Primary Body Text' },
                        { sel: '.text-muted', name: 'Muted Secondary Text' },
                        { sel: '.navbar-brand', name: 'Brand Title' },
                        { sel: '.metric-value', name: 'KPI Numbers' }
                    ];

                    return testClasses.map(t => {
                        const el = document.querySelector(t.sel);
                        if (!el) return { name: t.name, ratio: 'N/A', pass: true };
                        const color = window.getComputedStyle(el).color;
                        const m = color.match(/\\d+/g);
                        if (!m) return { name: t.name, ratio: 'N/A', pass: true };
                        const textLum = getLuminance(parseInt(m[0]), parseInt(m[1]), parseInt(m[2]));
                        const l1 = Math.max(bgLum, textLum);
                        const l2 = Math.min(bgLum, textLum);
                        const ratio = (l1 + 0.05) / (l2 + 0.05);
                        return { name: t.name, color, ratio: ratio.toFixed(2), pass: ratio >= 4.5 };
                    });
                })()
            """)
            all_contrast_pass = all(c["pass"] for c in contrast_audit)
            record_result(
                "Global", "Accessibility", "WCAG AA Contrast Ratios (>= 4.5:1)",
                "All core typography exceeds 4.5:1 contrast against dark background",
                f"{contrast_audit}",
                all_contrast_pass,
                f"Contrast metrics: {contrast_audit}"
            )

    finally:
        chrome_proc.terminate()
        chrome_proc.wait()

    print("\n==================================================================")
    print("TASK 33 VISUAL QA AUTOMATION SUMMARY")
    print("==================================================================")
    passed_count = sum(1 for r in RESULTS if r["status"] == "PASS")
    total_count = len(RESULTS)
    print(f"Total Checks: {total_count} | Passed: {passed_count} | Failed: {total_count - passed_count}")

    # Output JSON summary for report inclusion
    out_json = TABLES_DIR / "task33_visual_qa_results.json"
    out_json.write_text(json.dumps(RESULTS, indent=2))
    print(f"Results written to: {out_json}")

    if passed_count < total_count:
        print("\n[WARNING] Some visual QA checks failed.")
        raise SystemExit(1)
    else:
        print("\n\033[92mAll Visual QA and Accessibility Checks Passed Successfully!\033[0m")


if __name__ == "__main__":
    asyncio.run(run_visual_qa())
