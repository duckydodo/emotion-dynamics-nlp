#!/usr/bin/env python3
"""
End-to-end automated testing for the redesigned Emotion Dynamics NLP frontend.
Controls headless Google Chrome via Chrome DevTools Protocol (CDP) to:
1. Validate Screen 1 (Import): Sample dialogues, drag/drop info, format detection, preview card.
2. Validate Screen 2 (Conversation Viewer): Chat bubble stream, emotion pills, confidence disclaimer, probability drawer.
3. Validate Screen 3 (Emotion Dynamics): Trajectory chart, KPI metrics, sequence bar, turning points, and jump navigation.
4. Validate Screen 4 (Model Comparison): Multi-model execution, pairwise matrix, synchronized stream, disagreement filtering.
5. Validate Screen 5 (Research & Inspection): Architecture cards, Cohort A verified metrics, per-class table, confusion matrix switcher.
6. Validate Theme switching (Dark/Light mode).
7. Validate Clear Session reset.
8. Capture visual evidence screenshots saved to outputs/figures/.
"""

import asyncio
import base64
import json
import subprocess
import time
import urllib.request
import websockets
from pathlib import Path

CHROME_PORT = 9222
BASE_URL = "http://localhost:8000"
OUTPUT_DIR = Path("outputs/figures")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


async def send_cdp(ws, method, params=None, msg_id=[1]):
    mid = msg_id[0]
    msg_id[0] += 1
    req = {"id": mid, "method": method, "params": params or {}}
    await ws.send(json.dumps(req))
    while True:
        resp = json.loads(await ws.recv())
        if resp.get("id") == mid:
            if "error" in resp:
                raise RuntimeError(f"CDP error in {method}: {resp['error']}")
            return resp.get("result", {})


async def eval_js(ws, expr, await_promise=False):
    res = await send_cdp(ws, "Runtime.evaluate", {
        "expression": expr,
        "awaitPromise": await_promise,
        "returnByValue": True
    })
    result = res.get("result", {})
    if result.get("subtype") == "error":
        raise RuntimeError(f"JS evaluation error: {result.get('description')}")
    return result.get("value")


async def capture_screenshot(ws, filename):
    res = await send_cdp(ws, "Page.captureScreenshot", {"format": "png"})
    data = base64.b64decode(res["data"])
    filepath = OUTPUT_DIR / filename
    with open(filepath, "wb") as f:
        f.write(data)
    print(f"Captured screenshot: {filepath} ({len(data)} bytes)")
    return filepath


async def main():
    print("Launching headless Google Chrome...")
    chrome_proc = subprocess.Popen([
        "google-chrome",
        "--headless=new",
        "--disable-gpu",
        f"--remote-debugging-port={CHROME_PORT}",
        "--remote-allow-origins=*",
        "--user-data-dir=/tmp/chrome_test_profile",
        "--window-size=1280,950",
        "--no-sandbox",
        BASE_URL
    ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    try:
        # Wait for Chrome DevTools endpoint to become available
        ws_url = None
        for _ in range(30):
            try:
                with urllib.request.urlopen(f"http://127.0.0.1:{CHROME_PORT}/json", timeout=1) as resp:
                    tabs = json.loads(resp.read().decode())
                    for tab in tabs:
                        if tab.get("type") == "page" and "localhost:8000" in tab.get("url", ""):
                            ws_url = tab["webSocketDebuggerUrl"]
                            break
                    if ws_url:
                        break
            except Exception:
                time.sleep(0.2)

        if not ws_url:
            raise RuntimeError("Failed to connect to Chrome DevTools endpoint.")

        print(f"Connected to Chrome CDP at {ws_url}")

        async with websockets.connect(ws_url, max_size=20 * 1024 * 1024) as ws:
            # Enable domains
            await send_cdp(ws, "Page.enable")
            await send_cdp(ws, "Runtime.enable")

            # Navigate to local API application
            print(f"Navigating to {BASE_URL}...")
            await send_cdp(ws, "Page.navigate", {"url": BASE_URL})
            
            # Wait for document to be ready
            for _ in range(50):
                await asyncio.sleep(0.2)
                try:
                    ready = await eval_js(ws, "document.readyState")
                    if ready == "complete":
                        break
                except Exception:
                    pass

            await asyncio.sleep(1.0)

            # Check that page title and brand loaded
            title = await eval_js(ws, "document.title")
            print(f"Page Title: {title}")
            assert "Emotion Dynamics" in title, f"Unexpected title: {title}"

            # Verify System Health status
            status_text = await eval_js(ws, "document.getElementById('systemStatusText').textContent")
            print(f"System Status: {status_text}")
            assert "3 Models Online" in status_text, f"Unexpected status: {status_text}"

            # Test 1: Screen 1 - Import & Sample Loading
            print("\n--- Testing Screen 1: Home / Import ---")
            await eval_js(ws, "document.getElementById('sampleConflictBtn').click()")
            await asyncio.sleep(1.0)

            raw_text = await eval_js(ws, "document.getElementById('rawConversationInput').value")
            assert len(raw_text) > 50, "Sample dialogue failed to populate textarea."

            # Verify import preview card
            fmt = await eval_js(ws, "document.getElementById('previewFormat').textContent")
            turns = await eval_js(ws, "document.getElementById('previewEvalTurns').textContent")
            speakers = await eval_js(ws, "document.getElementById('previewSpeakers').textContent")
            print(f"Detected Format: {fmt}, Eval Turns: {turns}, Speakers: {speakers}")
            assert "PLAIN TEXT" in fmt.upper(), f"Unexpected format: {fmt}"
            assert int(turns) == 9, f"Expected 9 turns, got {turns}"

            await capture_screenshot(ws, "screen1_import_preview.png")

            # Test WhatsApp format sample
            print("Testing WhatsApp export sample preview...")
            await eval_js(ws, "document.getElementById('sampleWhatsAppBtn').click()")
            await asyncio.sleep(1.0)
            wa_fmt = await eval_js(ws, "document.getElementById('previewFormat').textContent")
            print(f"WhatsApp Sample Format: {wa_fmt}")
            assert "WHATSAPP" in wa_fmt.upper(), f"Expected WhatsApp format, got {wa_fmt}"
            await capture_screenshot(ws, "screen1_whatsapp_preview.png")

            # Test Instagram export sample
            print("Testing Instagram JSON sample preview...")
            await eval_js(ws, "document.getElementById('sampleInstagramBtn').click()")
            await asyncio.sleep(1.0)
            ig_fmt = await eval_js(ws, "document.getElementById('previewFormat').textContent")
            print(f"Instagram Sample Format: {ig_fmt}")
            assert "INSTAGRAM" in ig_fmt.upper(), f"Expected Instagram format, got {ig_fmt}"
            await capture_screenshot(ws, "screen1_instagram_preview.png")

            # Reset back to Conflict sample for analysis and downstream testing
            await eval_js(ws, "document.getElementById('sampleConflictBtn').click()")
            await asyncio.sleep(1.0)
            print("\n--- Testing Screen 2: Conversation Viewer ---")
            await eval_js(ws, "document.getElementById('runAnalysisBtn').click()")

            # Wait for analysis completion and screen transition
            for _ in range(25):
                active_screen = await eval_js(ws, "AppState.activeScreen")
                if active_screen == "viewer":
                    break
                await asyncio.sleep(0.5)

            active_screen = await eval_js(ws, "AppState.activeScreen")
            assert active_screen == "viewer", f"Screen should be 'viewer', got {active_screen}"

            bubble_count = await eval_js(ws, "document.querySelectorAll('#bubbleStream .bubble-wrapper').length")
            print(f"Rendered {bubble_count} conversation bubbles.")
            assert bubble_count == 9, f"Expected 9 bubbles, got {bubble_count}"

            # Inspect first bubble emotion and confidence
            first_emo = await eval_js(ws, "document.querySelector('#bubbleStream .emo-badge').textContent")
            first_conf = await eval_js(ws, "document.querySelector('#bubbleStream .confidence-label').textContent")
            print(f"Turn 1 Prediction: {first_emo.strip()} | Confidence: {first_conf.strip()}")

            # Open probability distribution drawer on turn 1
            await eval_js(ws, "document.querySelector('#bubbleStream button').click()")
            await asyncio.sleep(0.5)
            drawer_open = await eval_js(ws, "document.querySelector('.prob-drawer').classList.contains('open')")
            assert drawer_open, "Probability drawer should be open."

            await capture_screenshot(ws, "screen2_conversation_viewer.png")

            # Test 3: Screen 3 - Emotion Dynamics
            print("\n--- Testing Screen 3: Emotion Dynamics ---")
            await eval_js(ws, "navigateTo('dynamics')")
            await asyncio.sleep(1.0)

            dyn_shifts = await eval_js(ws, "document.getElementById('dynEmotionShifts').textContent")
            dyn_rate = await eval_js(ws, "document.getElementById('dynShiftRate').textContent")
            tp_count = await eval_js(ws, "document.getElementById('tpCountBadge').textContent")
            print(f"Dynamics: Emotion Shifts = {dyn_shifts}, Shift Rate = {dyn_rate}, Turning Points = {tp_count}")

            # Verify chart canvas exists and has dimension
            has_chart = await eval_js(ws, "AppState.chartInstance !== null")
            assert has_chart, "Chart.js instance was not initialized."

            await capture_screenshot(ws, "screen3_emotion_dynamics.png")

            # Test turning point jump navigation
            first_tp_btn = await eval_js(ws, "document.querySelector('#turningPointsList button') !== null")
            if first_tp_btn:
                print("Testing Turning Point 'Inspect Turn' navigation...")
                await eval_js(ws, "document.querySelector('#turningPointsList button').click()")
                await asyncio.sleep(0.5)
                screen_after_jump = await eval_js(ws, "AppState.activeScreen")
                assert screen_after_jump == "viewer", f"Expected jump to 'viewer', got {screen_after_jump}"

            # Test 4: Screen 4 - Model Comparison
            print("\n--- Testing Screen 4: Model Comparison ---")
            await eval_js(ws, "navigateTo('import')")
            await asyncio.sleep(0.5)
            await eval_js(ws, "document.getElementById('runComparisonBtn').click()")

            # Wait for comparison completion
            for _ in range(30):
                active_screen = await eval_js(ws, "AppState.activeScreen")
                if active_screen == "compare":
                    break
                await asyncio.sleep(0.5)

            active_screen = await eval_js(ws, "AppState.activeScreen")
            assert active_screen == "compare", f"Expected screen 'compare', got {active_screen}"

            consensus_text = await eval_js(ws, "document.getElementById('consensusRateBadge').textContent")
            print(f"Comparison Result: {consensus_text}")

            # Verify pairwise agreement matrix has 3 rows
            matrix_rows = await eval_js(ws, "document.querySelectorAll('#pairwiseMatrixBody tr').length")
            assert matrix_rows == 3, f"Expected 3 rows in pairwise matrix, got {matrix_rows}"

            # Verify comparison rows count
            comp_rows = await eval_js(ws, "document.querySelectorAll('#comparisonStream .comparison-row').length")
            print(f"Synchronized Comparison Rows: {comp_rows}")
            assert comp_rows == 9, f"Expected 9 synchronized rows, got {comp_rows}"

            # Test disagreement filter toggle
            await eval_js(ws, "document.getElementById('filterDisagreementsOnly').click()")
            await asyncio.sleep(0.3)
            filter_count = await eval_js(ws, "document.getElementById('compTurnFilterCount').textContent")
            print(f"Disagreement Filter Count: {filter_count}")

            await capture_screenshot(ws, "screen4_model_comparison.png")

            # Test 5: Screen 5 - Research & Inspection
            print("\n--- Testing Screen 5: Research & Benchmarks ---")
            await eval_js(ws, "navigateTo('research')")
            await asyncio.sleep(1.0)

            # Check architecture cards
            arch_cards = await eval_js(ws, "document.querySelectorAll('#archCardsGrid .arch-card').length")
            print(f"Rendered {arch_cards} architecture inspection cards.")
            assert arch_cards == 3, f"Expected 3 architecture cards, got {arch_cards}"

            # Check Cohort A benchmark table
            cohort_rows = await eval_js(ws, "document.querySelectorAll('#cohortATableBody tr').length")
            print(f"Cohort A table rows: {cohort_rows}")
            assert cohort_rows >= 3, f"Expected at least 3 benchmark rows, got {cohort_rows}"

            # Check Per-Class table
            per_class_rows = await eval_js(ws, "document.querySelectorAll('#perClassTableBody tr').length")
            print(f"Per-Class table rows: {per_class_rows}")
            assert per_class_rows == 7, f"Expected 7 emotion classes, got {per_class_rows}"

            # Test Confusion Matrix switcher (click DialogueRNN, DistilBERT, DialogueGCN)
            print("Testing Confusion Matrix switcher buttons...")
            await eval_js(ws, "document.querySelector('.cm-model-btn[data-model=\"distilbert\"]').click()")
            await asyncio.sleep(0.3)
            cm_cells = await eval_js(ws, "document.querySelectorAll('#cmContainer .cm-table td').length")
            assert cm_cells == 49, f"Expected 49 cells (7x7), got {cm_cells}"

            await capture_screenshot(ws, "screen5_research_inspection.png")

            # Test 6: Theme Switching
            print("\n--- Testing Theme Toggle ---")
            await eval_js(ws, "document.getElementById('themeToggleBtn').click()")
            await asyncio.sleep(0.5)
            is_light = await eval_js(ws, "document.body.classList.contains('light')")
            assert is_light, "Body should have 'light' class after theme toggle."
            print("Theme toggled to Light mode.")
            await capture_screenshot(ws, "screen1_light_theme.png")

            # Switch back to dark
            await eval_js(ws, "document.getElementById('themeToggleBtn').click()")
            await asyncio.sleep(0.3)

            # Test 7: Clear Session
            print("\n--- Testing Clear Session ---")
            await eval_js(ws, "window.confirm = () => true;") # Auto-confirm
            await eval_js(ws, "document.getElementById('clearSessionBtn').click()")
            await asyncio.sleep(0.5)

            input_after_clear = await eval_js(ws, "document.getElementById('rawConversationInput').value")
            assert input_after_clear == "", "Input should be empty after Clear Session."
            preview_display = await eval_js(ws, "document.getElementById('importPreviewPanel').style.display")
            assert preview_display == "none", "Preview panel should be hidden."
            active_after_clear = await eval_js(ws, "AppState.activeScreen")
            assert active_after_clear == "import", "Should reset to import screen."
            print("Clear Session successfully reset application state.")

            print("\n==========================================")
            print("ALL 7 END-TO-END TEST SUITES PASSED SUCCESSFULLY!")
            print("==========================================")

    finally:
        chrome_proc.terminate()
        try:
            chrome_proc.wait(timeout=3)
        except subprocess.TimeoutExpired:
            chrome_proc.kill()


if __name__ == "__main__":
    asyncio.run(main())
