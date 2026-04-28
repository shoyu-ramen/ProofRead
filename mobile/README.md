# ProofRead — Mobile (v1)

Cross-platform Expo app that captures alcoholic beverage labels and
verifies them against TTB compliance rules using the ProofRead backend.

This directory is the v1 scaffold: every screen from SPEC §v1.7 has a
file, the API client matches the contract in `backend/app/api/scans.py`,
and the camera screen is wired with a structural stub for the
Vision-Camera + frame-processor pipeline.

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
│           ├── camera/[surface].tsx
│           ├── review.tsx
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
│   │   └── scanStore.ts        # in-progress scan draft
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
| Frame-processor pre-checks | `PreCheck` state flips to `ready` after 600 ms. Real focus / glare / coverage / motion checks (SPEC §v1.5 F1.3) are TODO. | `app/(app)/scan/camera/[surface].tsx` |
| HDR / thermal-state adaptation | `takePhoto` uses defaults. SPEC §0.5 calls for HDR + adaptive frame-processor frequency. | `app/(app)/scan/camera/[surface].tsx` |
| Bbox-on-image overlay | Rule detail draws a schematic instead of the captured image with the bbox overlaid. Backend doesn't yet return image_id alongside bbox. | `app/(app)/scan/rule/[ruleId].tsx` |
| Recent scans / history list | Empty state — `GET /v1/scans` (history) endpoint not yet on the backend scaffold. | `app/(app)/home.tsx`, `app/(app)/history.tsx` |
| Image-retention persistence | UI selects a value but no PUT endpoint to save it. | `app/(app)/settings.tsx` |
| Flag rule result | API method exists; UI does not yet collect the comment. | `app/(app)/scan/rule/[ruleId].tsx`, `src/api/client.ts` |
| Cancel during processing | Navigates away client-side; backend has no cancel endpoint in v1. | `app/(app)/scan/processing/[id].tsx` |

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
- **Surface routing.** v1 captures only `front` + `back`. The data
  model allows `side` and `neck`; those are typed but not surfaced in
  the flow.
- **Gracefully degraded camera screen.** When permission is denied or
  the device has no back camera, the screen renders a fallback panel
  rather than blocking the entire flow.

## Type-check

```sh
cd mobile
npm install
npm run typecheck
```

`tsc --noEmit` is configured with `strict: true` plus the explicit
strict-family flags.
