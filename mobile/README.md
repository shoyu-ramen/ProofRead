# ProofRead — Mobile (v1)

Cross-platform Expo app that captures alcoholic beverage labels and
verifies them against TTB compliance rules using the ProofRead backend.

This directory is the v1 scaffold: every screen from SPEC §v1.7 has a
file, the API client matches the contract in `backend/app/api/scans.py`,
and the cylindrical scan screen runs a real Vision-Camera + frame-
processor pipeline that produces an unrolled-label panorama in one
rotation pass (see `.claude/CYLINDRICAL_SCAN_ARCHITECTURE.md`).

## Tech stack

Frozen by SPEC §0:

- Expo SDK 51 / React Native 0.74 / TypeScript (strict)
- expo-router (file-based navigation)
- react-native-vision-camera v4 (frame processors, photo capture)
- @tanstack/react-query (server state)
- zustand (local state)
- expo-secure-store (auth tokens — wired but not used yet)
- react-native-reanimated v3 (camera overlays — Babel plugin enabled)

## Setup

```sh
cd mobile
npm install              # or pnpm / yarn — only npm tested in CI
npx expo prebuild        # generates ios/ + android/ for vision-camera
npx expo run:ios         # iOS dev build (requires Xcode + simulator)
npx expo run:android     # Android dev build (requires Android SDK)
npm run typecheck        # tsc --noEmit
```

For Expo Go testing without native modules, `npx expo start` works for
every screen except the camera — Vision Camera v4 needs a custom dev
client.

### Backend

Point the app at your local backend by editing
`app.json → expo.extra.apiBaseUrl` (defaults to `http://localhost:8000`).
On a physical device use your machine's LAN IP, not `localhost`.

## Layout

```
mobile/
├── app/                        # expo-router routes
│   ├── _layout.tsx             # root: providers + Stack
│   ├── index.tsx               # splash / redirect
│   ├── signin.tsx              # stub auth
│   └── (app)/                  # authenticated group
│       ├── _layout.tsx
│       ├── home.tsx
│       ├── history.tsx
│       ├── settings.tsx
│       └── scan/
│           ├── beverage-type.tsx
│           ├── container-size.tsx
│           ├── unwrap.tsx           # cylindrical scan (live)
│           ├── review.tsx           # panorama preview + analyze
│           ├── processing/[id].tsx
│           ├── report/[id].tsx
│           └── rule/[ruleId].tsx
├── src/
│   ├── api/
│   │   ├── client.ts           # typed fetch wrapper
│   │   └── types.ts            # mirrors backend pydantic schemas
│   ├── state/
│   │   ├── auth.ts             # zustand fake-user store
│   │   ├── queryClient.ts      # tanstack-query setup
│   │   └── scanStore.ts        # in-progress scan draft (panorama-shaped)
│   ├── scan/                   # cylindrical-scan subsystems
│   │   ├── tracker/            # CV: bottle silhouette + optical flow + angle
│   │   ├── panorama/           # Skia: live unrolled-label canvas + stitcher
│   │   ├── ui/                 # silhouette / ring / dial / chip / reveal
│   │   ├── state/              # scan state machine (aligning → … → complete)
│   │   └── hooks/              # useScanStateMachine, useRotationCapture, useScanHaptics
│   ├── components/             # Button, Screen, ProgressBar, …
│   └── theme.ts                # color/spacing/typography tokens
├── app.json                    # Expo config (vision-camera plugin)
├── babel.config.js             # reanimated plugin
├── tsconfig.json               # strict TS + path aliases
├── package.json
└── README.md (this file)
```

## Status of stubs

Every TODO in source is grep-able — these are the major ones:

| Area | What's stubbed | Where |
|---|---|---|
| Auth0 sign-in | Single "Sign in (stub)" button populates a fake user. No token exchange yet. | `app/signin.tsx`, `src/state/auth.ts` |
| HDR / thermal-state adaptation | `takePhoto` uses defaults. SPEC §0.5 calls for HDR + adaptive frame-processor frequency. The tracker frame processor adapts stride under thermal load; HDR is still TODO. | `app/(app)/scan/unwrap.tsx`, `src/scan/tracker/frameProcessor.ts` |
| Bbox-on-image overlay | Rule detail renders the bbox over the local panorama; on a fresh device the panorama is missing so the overlay is skipped with a hint. | `app/(app)/scan/rule/[ruleId].tsx` |
| Recent scans / history list | Hidden until backend ships `GET /v1/scans`. | `app/(app)/home.tsx` |
| Image-retention persistence | UI selects a value but no PUT endpoint to save it. | `app/(app)/settings.tsx` |
| Flag rule result | API method exists; UI does not yet collect the comment. | `app/(app)/scan/rule/[ruleId].tsx`, `src/api/client.ts` |
| Cancel during processing | Navigates away client-side; backend has no cancel endpoint in v1. | `app/(app)/scan/processing/[id].tsx` |

## Cylindrical scan

The scan flow is a single cylindrical-rotation pass: the user holds
the bottle in front of the camera and rotates it once; the live
unrolled-label panorama paints into a Skia canvas at the top of the
screen as each angular checkpoint is captured. The plumbing is split
across three subsystems under `src/scan/`:

- `tracker/` — Vision Camera v4 frame processor. Resizes to 160×240,
  computes pre-check signals (blur / glare / coverage / motion),
  detects the bottle silhouette via vertical Sobel medians, measures
  horizontal optical flow against the previous frame, and integrates
  the angular position. Adapts stride under thermal load.
- `panorama/` — Skia off-screen surface. `extractStrip()` cuts a
  vertical column from each captured photo; the live `<PanoramaCanvas>`
  paints strips into the panorama as they arrive; `stitchPanorama()`
  encodes the final JPEG on completion.
- `ui/` — overlay components driven by Reanimated shared values
  (silhouette, rotation guide ring, progress dial, quality chip,
  cancel button, completion reveal). All visuals follow tokens in
  `src/theme.ts` (`scanGeometry`, `scanMotion`, the `scan*` color
  family).

The scan screen (`app/(app)/scan/unwrap.tsx`) hosts the camera, runs
the tracker, owns the state machine (`aligning → ready → scanning ↔
paused → complete | failed`), and triggers the checkpoint capture
queue. See `.claude/CYLINDRICAL_SCAN_ARCHITECTURE.md` for the full
contract.

## Decisions made

A few spec areas were ambiguous; the choices made here:

- **Splash screen.** SPEC §v1.7 lists a "Splash — Logo, 1s timeout"
  row. We treat `app/index.tsx` as the splash, redirecting based on
  auth state. The 1-second logo animation can be added by gating the
  redirect behind a small timer; the structural piece (the entry route)
  is in place.
- **Image overlay drawer on the report.** SPEC §v1.7 says "image
  overlay drawer". We picked a navigation-based pattern (tap a rule →
  open `scan/rule/[ruleId]`) over an inline modal drawer to keep the
  layout simple in v1. Both satisfy the spec.
- **`is_imported` toggle.** SPEC §v1.6 doesn't show this on the
  container-size step, but the backend's `CreateScanRequest` accepts
  `is_imported` and the `country_of_origin.presence_if_imported` rule
  needs it. The toggle lives on the container-size step.
- **Gracefully degraded camera screen.** When permission is denied or
  the device has no back camera, the scan screen renders a fallback
  panel rather than blocking the entire flow.

## Type-check

```sh
cd mobile
npm install
npm run typecheck
```

`tsc --noEmit` is configured with `strict: true` plus the explicit
strict-family flags.
