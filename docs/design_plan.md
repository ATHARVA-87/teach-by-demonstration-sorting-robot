# Design plan: operator console + AR teaching overlay

Status: **approved with revisions (D6, 2026-10-25)**. This is revision 2, with the 13 required revisions applied. Process: frontend-design skill (grounding → plan → review against generic defaults → build → critique).

## 1. Grounding
- **Product:** operator console for an industrial sorting cell that workers teach by demonstration (application to CrossCompass / Satake).
- **Audience:** robotics engineers watching a demo video at 1080p, with the MuJoCo viewer beside the console.
- **Primary job:** make **teach → learned rule → robot execution** readable at a glance.
- **References from the subject's world:**
  - **Inspection systems** on sorting lines: the camera image is the main surface; regions of interest drawn as **corner brackets**; measurements as small callouts; pass/fail as a large binary indicator.
  - **ISA-101 high-performance HMI:** grey base; color reserved for running status and abnormal states; values shown against limits.
  - **Machine control panels:** a mode selector (one mode at a time, always labeled), a cell-state annunciator, node lamps, and an ISO 13850 emergency stop.

### The central rule: item color is data, UI color is state
In this cell the item colors (red / green / blue / yellow) **are the data being sorted**.
- **Item paint** appears only on **item glyphs**: in the AR view, in the robot camera view, and on item chips in panels. A chip always shows a **shape glyph plus a color word** ("red cube small"), so identity never depends on color alone.
- **Status hues** (green, yellow, red) **never** appear inside the two image views or on an item glyph.
- **In the chrome,** status hues may appear only in:
  - the cell-state annunciator,
  - the node lamps,
  - the e-stop,
  - log severity text, which always carries a word prefix ("Warning:", "Error:").
- Boxes are identified by **letter + fill pattern**, never by color.

## 2. Color tokens
| Name | Hex | Role |
|---|---|---|
| Black grey | `#2E3236` | page background (RAL 7021 "Black grey") |
| Anthracite | `#383E42` | panels, trays, gauge/bar tracks (RAL 7016 "Anthracite grey") |
| Signal white | `#ECECE7` | text, outlines, brackets, hand cursor, data bars (RAL 9003) |
| Run green | `#3DAA5C` | fill only: cell state Ready/Running; healthy lamps |
| Signal yellow | `#F2B70F` | fill only: Paused/Teaching-attention, warning lamps, e-stop backplate |
| Signal red | `#D23A2E` | the e-stop actuator only (always on its yellow backplate) |

Derived steps (same hues, contrast computed with WCAG):
| Step | Hex | Use | Contrast |
|---|---|---|---|
| Ink | `#1E2124` | all text on green or yellow fills | 5.47:1 / 8.9:1 |
| Fault fill | `#B8281F` | Fault annunciator, with Signal-white text | 5.27:1 |
| Error text | `#FF8A7A` | error log text on dark | 5.64:1 on Black grey, 4.74:1 on Anthracite |
| Green text | `#52C474` | if green is ever needed as text | 4.91:1 on Anthracite |
| Secondary text | `#A7A9A8` | only on Anthracite or darker | 4.59:1 minimum |
| Divider | `#474D52` | 1 px rules |

Banned pairs: white on yellow (1.53:1), white on Run green (2.49:1), Signal red on bare Black grey (2.70:1).

**Item paint** is not a UI token. It's defined once in `sorting_scene/config/scene.yaml` and used identically by the AR renderer and MuJoCo (digital twin).

## 3. Type
One family, two widths, bundled (SIL OFL) under `sorting_ws/src/sorting_common/assets/fonts`, so it works offline:
- **Barlow Semi Condensed 600:** cell state (36 px), mode names, numbers (tabular figures, `tnum`), the rule sentence in Review (28 px).
- **Barlow 400/500:** body, tables, event log (14–15 px).
- **AR overlay:** Barlow Semi Condensed via Pillow, with a dark halo.
- **Symbols** (arrows, checkmarks, step separators, shape glyphs) are **never font glyphs**: the Barlow cmaps lack → ▸ ✓ ✗ ■ ● ▲ ⌜. They're inline SVG on the web and Pillow/OpenCV primitives in the AR overlay.
- Sentence case everywhere; no all-caps labels; no monospace.

## 4. Layout
Target viewport **1280 × 940**: a 1920 × 1080 screen minus the Ubuntu top bar (~32 px) and browser chrome (~85 px); the MuJoCo viewer takes the remaining ~620 px of width. To be verified with a headless screenshot at 1280 × 940 when the console is built (M6).

**Frame (always visible):**
```
┌──────────────────────────────────────────────────────────────────────────────┐
│ Sorting cell   [Teach][Review][Run][Results]   ┌─────────────────┐ ┌──────┐ │
│                                                 │ Cell: Teaching  │ │ (e)  │ │
│                                                 └─────────────────┘ │Emerg.│ │
│ lamps: Camera OK · Hand tracking OK · Rule learner OK · Simulator —  │ stop │ │
│        Perception — · Executor — · Safety OK                         └──────┘ │
├──────────────────────────────────────────────────────────────────────────────┤
│                         mode content (below)                                 │
└──────────────────────────────────────────────────────────────────────────────┘
```
- The annunciator shows one of the spec's six states: Idle / Teaching / Ready / Running / Paused / Fault.
- **Lamps:** a 10 px dot plus a text state ("OK", "No heartbeat 3 s", "Not started"); green/yellow dots with words, never color alone. (Middle dots in the sketch stand for separate labeled fields.)
- **E-stop:** a 64 px red mushroom on a yellow backplate, labeled "Emergency stop".

**Latched e-stop / fault state:**
```
┌──────────────────────────────────────────────────────────────────────────────┐
│ Sorting cell   [Teach][Review][Run][Results]  ┌──────────────────────────┐   │
│                (disabled: "Reset the cell    │ Fault                     │   │
│                 to change mode")             │ Emergency stop pressed.   │(e)│
│                                              │ Check the cell, then      │   │
│                                              │ press Reset.   [Reset]    │   │
│                                              └──────────────────────────┘   │
```
E-stop and watchdog trips both go to **Fault**. Leaving Fault needs an explicit **Reset**, after which the cell goes to Idle; a running task does not resume by itself.

**Teach:**
```
┌───────────────────────────────────────────┬──────────────────────────────────┐
│ AR teaching view (webcam + overlay)       │ Candidate rules                  │
│ 4:3, ~720 × 540                           │  Sort by color      [████░] 62%  │
│                                           │  Large to B, else A [██░░░] 21%  │
│                                           │  (white bars, Anthracite track)  │
│                                           ├──────────────────────────────────┤
│                                           │ Question (only while asking)     │
│                                           │  Where does this item go?        │
│                                           │  [chip: blue prism large]        │
│                                           │  Pick it up in the view and drop │
│                                           │  it in a box.                    │
├───────────────────────────────────────────┴──────────────────────────────────┤
│ Demonstrations  #1 [chip red cube small] ⟶ A  Remove   #2 [chip…] ⟶ B  Remove  │
│ Note: #2 and #5 conflict: same kind of item, different boxes.                │
│       Remove one, or sort another to decide.                                 │
└──────────────────────────────────────────────────────────────────────────────┘
```
(⟶ = SVG arrow.)

**Review:**
```
┌──────────────────────────────────────────────────────────────────────────────┐
│ Learned rule                                                  Confidence 97%  │
│   Sort by color: red to A, blue to B, others to C                             │
│                                                                              │
│ Evidence                                   Competing rules                    │
│  [svg check] #1 red cube small to A        Large to B, else A       2%   (o)  │
│  [svg check] #2 blue cylinder large to B   Shape: cube to A, …      1%   (o)  │
│  [svg cross] #5 red prism large to B:                                         │
│              conflicts with #1                                                │
│                                                                              │
│  [Approve rule]  [Correct rule]  [Teach more]                                │
│   primary        secondary       secondary                                   │
│                                                                              │
│ Correct rule (expands): pick a competing rule (radio), or edit the mapping:  │
│   value   red [A ▾]  blue [B ▾]  green [C ▾]  yellow [C ▾]   [Save rule]      │
│                                                                              │
│ After approve: "Saved task file tasks/sort_by_color_2026-10-26.json"          │
└──────────────────────────────────────────────────────────────────────────────┘
```
- **Primary button:** Signal-white fill with ink text.
- **Secondary button:** 1 px Signal-white outline.

**Run:**
```
┌──────────────────────────────────────────────────────────────────────────────┐
│ Rule: Sort by color: red to A, blue to B, others to C                        │
│ Task file: tasks/sort_by_color.json        [Start] [Pause] [Resume]           │
├───────────────────────────────────────────┬──────────────────────────────────┤
│ Robot camera (detections: white corner     │ Step                             │
│ brackets with dark halo; callout fields:   │ Perceive > Classify > Apply rule │
│ "red", "cube", "small", "94%")             │ > Plan > Pick > Verify > Place   │
│                                           │ (current lit; sub-state below)   │
│                                           │ Retry grasp 2 of 3               │
│                                           ├──────────────────────────────────┤
│                                           │ Sorted   Skipped   Failed        │
│                                           │   7         1         0          │
│                                           │ Throughput 5.8 items/min         │
├───────────────────────────────────────────┴──────────────────────────────────┤
│ Decisions  item 3 [chip] red cube small: rule "red to A": box A: placed       │
│            item 5 [chip]: unreachable: skipped (reported)                    │
├──────────────────────────────────────────────────────────────────────────────┤
│ Events   23:41:02  Warning: grasp slipped on item 4. Retrying.               │
└──────────────────────────────────────────────────────────────────────────────┘
```

**Results:**
```
┌──────────────────────────────────────────────────────────────────────────────┐
│ Rule learning            Perception              Execution                   │
│  4.1 demos (active)       96.8% attribute acc.     97.5% pick success        │
│  7.9 demos (random)       2.1 mm mean loc. error   6.2 items/min             │
│  [plot]                   [confusion matrix]       [plot]                    │
│                                                                              │
│ Empty state: "No benchmark results yet. Run the benchmark launch file to     │
│ create them."                                                                │
└──────────────────────────────────────────────────────────────────────────────┘
```
(Numbers in the sketch are placeholders. Real values come only from benchmark CSVs.)
- **Plots:** Signal white solid vs `#A7A9A8` dashed, with distinct markers; the confusion matrix uses a grey→white ramp; never status colors.
- The scripts also write a light version for the README and the report.

## 5. AR overlay (drawn on the webcam frame)
Every Signal-white stroke (brackets, cursor ring, pinch arc, tray outline, text) gets a **1 px dark halo**, so it reads against white walls.

| Element | Drawing |
|---|---|
| Table area | Black-grey scrim at 35% + 1 px Signal-white border with halo |
| Items | Flat chips in the item's scene paint: square = cube, circle = cylinder, triangle = prism. Drawn to scale (large ≥ 1.5× small); 2 px dark outline |
| Boxes | Open trays: Anthracite fill + pattern by slot index (A diagonal hatch, B dots, C plain, D cross-hatch, E vertical lines, then repeat) + large letter. The same letter and pattern on the MuJoCo boxes |
| Hover | Signal-white corner brackets around the item under the pinch point |
| Grab | Brackets thicken; the chip lifts with a small dark offset shadow |
| Drop target | The tray under a held item gets a 3 px outline and an inverted letter (ink on a white tile) |
| Hand cursor | Signal-white ring at the pinch point; filled when pinched; an arc shows pinch strength |
| Question item | Signal-white **dashed** brackets + a "?" tag on a dark scrim |
| Messages | One banner at the top on a dark scrim, Signal-white text |

**Copy:**
- "Too dark to track your hand. Turn on a light or face a window."
- "Hand lost. Item returned to its place."
- "Dropped between boxes. Item returned to the table."
- "No webcam found. Plug one in or close other apps using it."

## 6. Interaction and motion
- **One motion moment:** the annunciator's fill change (150 ms). No entrance animations. `prefers-reduced-motion` makes it instant.
- **E-stop:** click, Space/Enter, or global `Esc`. Release needs an explicit **Reset** (two steps).
- **Mode selector:** buttons with `aria-pressed`. Modes that aren't available are disabled with a reason ("Approve a rule before running").
- **Focus:** 3 px Signal-white ring with offset.
- **Errors:** say what happened and how to fix it; no apologies.

## 7. Principles
1. **Item color is data; UI color is state.** The two never mix (section 1).
2. **The image is the instrument.** Camera views are the largest surfaces; text annotates them, like an inspection system.
3. **One bold object: the e-stop.** Everything else is quiet: no shadows except the lifted item, 4 px radii, 1 px dividers.
4. **Show the reasoning.** Rules appear with their evidence, conflicts and competitors.
5. **Legible in a 1080p recording:** body ≥ 14 px, key numbers ≥ 28 px, text contrast ≥ 4.5:1, non-text ≥ 3:1.

## 8. Review against the skill's generic defaults
| Default | Result |
|---|---|
| Near-black + single acid accent | Pass: RAL Black grey / Anthracite; no decorative accent |
| SaaS card kit | Pass: panels differ by function; no shadows |
| Inter / default sans | Pass: Barlow |
| Monospace data labels | Pass: tabular figures |
| All-caps eyebrows, "WORD — fragment", middle-dot strings | Pass: sentence case, labeled fields |
| Scattered entrance animations | Pass: one transition |
| Arrows appended to buttons | Pass: SVG arrows only where data flows |
| Font glyphs the bundled font lacks | Fixed in rev 2 (SVG / primitives) |
| Color rule violations (yellow question marker, red conflict text) | Fixed in rev 2 |
