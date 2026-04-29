/**
 * Live unrolled-panorama canvas (ARCH §5.2).
 *
 * Maintains a single off-screen Skia surface sized to the full
 * panorama (~2880×1024) and stamps each new StripCheckpoint into it
 * exactly once. The visible <Canvas> only needs to draw the surface's
 * current snapshot scaled to fit its panel — we never re-paint old
 * strips, which keeps the per-frame cost flat regardless of how many
 * strips have landed.
 *
 * Visual treatment per ARCH §5.2 / SCAN_DESIGN.md:
 *   - 200ms opacity ramp on each new strip's draw
 *   - vertical "writing edge" cursor at the next-strip x position
 *   - subtle pulse on every strip arrival
 *   - dot-pattern fill where unfilled
 */

import React, {
  useEffect,
  useMemo,
  useRef,
} from 'react';
import { StyleSheet, Text, View, type StyleProp, type ViewStyle } from 'react-native';
import {
  AlphaType,
  Canvas,
  ColorType,
  Skia,
  Image as SkiaImage,
  Group,
  Rect,
  Fill,
  ImageShader,
} from '@shopify/react-native-skia';
import {
  Easing as ReanimatedEasing,
  useDerivedValue,
  useSharedValue,
  withTiming,
  type SharedValue,
} from 'react-native-reanimated';
import type {
  SkCanvas,
  SkImage,
  SkSurface,
} from '@shopify/react-native-skia';

import { colors, scanGeometry } from '@src/theme';
import {
  STRIP_WIDTH as DEFAULT_STRIP_WIDTH,
  coverageToStripX,
  type PanoramaState,
  type StripCheckpoint,
} from './types';

interface Props {
  state: PanoramaState;
  /**
   * Live snapshot of the off-screen panorama surface. Owned by the
   * parent so `stitchPanorama` can encode the already-painted bytes
   * directly instead of allocating a second surface and repainting
   * every strip — eliminates ~12 MB of peak memory at scan completion
   * and the second paint pass that came with it.
   *
   * The component writes a fresh `makeImageSnapshot()` here on every
   * strip arrival (and on init); previous snapshots are disposed so
   * the SharedValue holds at most one live SkImage.
   */
  snapshotSv: SharedValue<SkImage | null>;
  style?: StyleProp<ViewStyle>;
  /** Fired once per strip after it's painted into the off-screen surface. */
  onStripDrawn?: (idx: number) => void;
}

// Visual constants — motion timings stay inline; colors + dot geometry
// resolve to theme tokens (SCAN_DESIGN §3.2) so the empty strip reads
// at the intended contrast.
const STRIP_FADE_MS = 200;
const PULSE_MS = 320;
const WRITING_EDGE_WIDTH = scanGeometry.panoramaWritingEdgeWidth;
const WRITING_EDGE_COLOR = colors.panoramaWritingEdge;
const DOT_BG_COLOR = colors.panoramaBg;
const DOT_FG_COLOR = colors.panoramaEmptyDot;
const DOT_SIZE = scanGeometry.panoramaEmptyDotSize;
const DOT_SPACING = scanGeometry.panoramaEmptyDotPitch;

export function PanoramaCanvas({
  state,
  snapshotSv,
  style,
  onStripDrawn,
}: Props): React.ReactElement {
  const { width, height } = state;

  // The off-screen surface and its mutable backing image. We hold both
  // imperatively (refs) so we can mutate per checkpoint without forcing
  // a React re-render of unrelated state. `snapshotSv` is parent-owned
  // so the stitcher can read the already-painted bytes at completion
  // (see Props doc).
  const surfaceRef = useRef<SkSurface | null>(null);
  const lastDrawnIdxRef = useRef<number>(-1);

  // Audit finding: the RGB→RGBA widening pass inside `makeImageFromRgb`
  // was allocating a fresh Uint8Array per strip — at STRIP_WIDTH (80) ×
  // height (~1024) × 4 bytes ≈ 327680 bytes/strip, and ~36 strips/scan,
  // that's ~11.8MB of GC churn during the live scan window when the
  // tracker, capture queue, and stitcher are already competing for
  // bandwidth. Reusing a single buffer across the whole scan eliminates
  // that. Skia.Data.fromBytes copies the source bytes synchronously, so
  // overwriting our buffer for the next strip is safe.
  const rgbaBufferRef = useRef<Uint8Array | null>(null);

  // Reanimated drivers — opacity ramp on the most recent strip, and a
  // pulse on the panel that fires whenever a strip lands.
  const fadeSv = useSharedValue<number>(0);
  const pulseSv = useSharedValue<number>(0);
  const writingEdgeSv = useSharedValue<number>(0);

  // Allocate (and reallocate on dimension changes) the off-screen
  // surface. The surface is the heavyweight allocation; everything
  // else is incremental.
  useEffect(() => {
    const next = Skia.Surface.MakeOffscreen(width, height);
    if (!next) {
      // Surface allocation can fail under memory pressure. Render path
      // tolerates a null surface (falls back to dot-pattern only).
      return;
    }
    const canvas = next.getCanvas();
    paintDotBackground(canvas, width, height);
    next.flush();
    // Replace the previous snapshot atomically, disposing the old one
    // so we don't leak ~12MB per width/height change. react-native-skia
    // 1.x's makeImageSnapshot returns an SkImage that holds GPU memory;
    // assigning a new SkImage to the SharedValue without disposing the
    // previous one is the leak pattern documented in skia#2079/#2909.
    const oldSnapshot = snapshotSv.value;
    snapshotSv.value = next.makeImageSnapshot();
    if (oldSnapshot && typeof oldSnapshot.dispose === 'function') {
      oldSnapshot.dispose();
    }
    surfaceRef.current = next;
    lastDrawnIdxRef.current = -1;

    return () => {
      const finalSnap = snapshotSv.value;
      snapshotSv.value = null;
      if (finalSnap && typeof finalSnap.dispose === 'function') {
        finalSnap.dispose();
      }
      if (typeof next.dispose === 'function') next.dispose();
      surfaceRef.current = null;
    };
  }, [width, height, snapshotSv]);

  // Whenever new strips arrive, paint the suffix into the surface and
  // kick the fade/pulse animations.
  useEffect(() => {
    const surface = surfaceRef.current;
    if (!surface) return;

    const last = lastDrawnIdxRef.current;
    const next = state.strips.length - 1;
    if (next <= last) return;

    const canvas = surface.getCanvas();
    for (let i = last + 1; i <= next; i++) {
      const strip = state.strips[i];
      paintStripIntoSurface(canvas, strip, width, rgbaBufferRef);
      onStripDrawn?.(i);
    }
    surface.flush();
    // Same disposal dance as the init path — drop the previous SkImage
    // before publishing the new one. Without this we leak one SkImage
    // per strip (≈12MB at 2880×1024 RGBA × ~36 strips per scan).
    const prevSnap = snapshotSv.value;
    snapshotSv.value = surface.makeImageSnapshot();
    if (prevSnap && typeof prevSnap.dispose === 'function') {
      prevSnap.dispose();
    }
    lastDrawnIdxRef.current = next;

    // Animate: fade ramp on the newest strip, panel pulse, writing-edge
    // catch-up. Reset → drive avoids stale interpolation on rapid
    // checkpoints.
    fadeSv.value = 0;
    fadeSv.value = withTiming(1, {
      duration: STRIP_FADE_MS,
      easing: ReanimatedEasing.out(ReanimatedEasing.cubic),
    });
    pulseSv.value = 0;
    pulseSv.value = withTiming(1, {
      duration: PULSE_MS,
      easing: ReanimatedEasing.out(ReanimatedEasing.quad),
    });
    const newCoverage = state.strips[next]?.coverage ?? 0;
    writingEdgeSv.value = withTiming(newCoverage, {
      duration: STRIP_FADE_MS,
      easing: ReanimatedEasing.out(ReanimatedEasing.cubic),
    });
  }, [state.strips, width, onStripDrawn, fadeSv, pulseSv, writingEdgeSv, snapshotSv]);

  // Skia derived value for the writing-edge x-coordinate. We bridge
  // from the Reanimated shared value so this stays on the UI thread.
  // Using the canonical STRIP_WIDTH constant (rather than a divisor
  // derived from the live strip count) avoids the cursor pinning at
  // x=0 for the first strip — the strip-count divisor produced
  // `width / 1 == width`, which made `coverageToStripX` collapse to 0.
  const writingEdgeX = useDerivedValue(() => {
    const cov = writingEdgeSv.value;
    return coverageToStripX(cov, width, DEFAULT_STRIP_WIDTH);
  }, [writingEdgeSv, width]);

  // Pulse opacity: starts at 0.35, decays to 0 across PULSE_MS.
  const pulseOpacity = useDerivedValue(() => {
    return 0.35 * (1 - pulseSv.value);
  }, [pulseSv]);

  // Layout: the parent panel decides actual size; we render the Canvas
  // full-bleed with the full panorama scaled to fit via Skia's image
  // sampler.
  const containerStyle = useMemo(
    () => [styles.container, style],
    [style],
  );

  const showLegend = state.strips.length === 0;

  return (
    <View style={containerStyle} accessibilityRole="image" accessibilityLabel="Live unrolled bottle label">
      <Canvas style={StyleSheet.absoluteFill}>
        {/* Background dots, drawn behind the panorama snapshot so the
            "not yet captured" regions of the surface read through. */}
        <Fill color={DOT_BG_COLOR} />
        <DotPattern width={width} height={height} />

        {/* The panorama snapshot itself, scaled to fit. Sk Image node
            accepts a SharedValue<SkImage|null>; rendering null is a
            no-op, so the dot background shows through during the
            single-frame surface init. */}
        <PanoramaSnapshot snapshot={snapshotSv} width={width} height={height} />

        {/* Writing-edge cursor — a thin vertical line at the next-strip x. */}
        <Rect
          x={writingEdgeX}
          y={0}
          width={WRITING_EDGE_WIDTH}
          height={height}
          color={WRITING_EDGE_COLOR}
        />

        {/* Pulse overlay — full-bleed white, fades to 0 after each strip. */}
        <Rect
          x={0}
          y={0}
          width={width}
          height={height}
          color="white"
          opacity={pulseOpacity}
        />
      </Canvas>
      {showLegend ? (
        <View pointerEvents="none" style={styles.legendWrap}>
          <Text style={styles.legendText} numberOfLines={1}>
            Your label appears here as you rotate
          </Text>
        </View>
      ) : null}
    </View>
  );
}

/**
 * SkiaImage node wrapper — exists so the snapshot SharedValue plumbs
 * cleanly into the declarative tree. Skia's <Image> accepts an
 * SkImage|null directly via shared values in 1.x.
 */
function PanoramaSnapshot({
  snapshot,
  width,
  height,
}: {
  snapshot: ReturnType<typeof useSharedValue<SkImage | null>>;
  width: number;
  height: number;
}): React.ReactElement | null {
  // The Image node re-renders whenever its image prop changes; the
  // shared value drives that on the UI thread.
  return (
    <SkiaImage
      image={snapshot}
      x={0}
      y={0}
      width={width}
      height={height}
      fit="fill"
    />
  );
}

/**
 * Static dot pattern — drawn once per layout into the parent's tree.
 * Cheap (a few hundred dots at most) and visually quiet.
 */
function DotPattern({
  width,
  height,
}: {
  width: number;
  height: number;
}): React.ReactElement {
  const cols = Math.ceil(width / DOT_SPACING);
  const rows = Math.ceil(height / DOT_SPACING);
  // We rely on Skia's <Group> + a single ImageShader for performance:
  // build a tiny tile shader and tile it across the panel. Falls back
  // to a sparse <Rect> grid if the shader can't be built.
  const tile = useMemo(() => buildDotTile(), []);
  if (tile) {
    return (
      <Group>
        <Rect x={0} y={0} width={width} height={height}>
          <ImageShader
            image={tile}
            tx="repeat"
            ty="repeat"
            fit="fill"
            rect={{ x: 0, y: 0, width: DOT_SPACING, height: DOT_SPACING }}
          />
        </Rect>
      </Group>
    );
  }
  // Fallback grid — used only on platforms where shader construction
  // fails at component init.
  const rects: React.ReactElement[] = [];
  for (let r = 0; r < rows; r++) {
    for (let c = 0; c < cols; c++) {
      rects.push(
        <Rect
          key={`d-${r}-${c}`}
          x={c * DOT_SPACING + (DOT_SPACING - DOT_SIZE) / 2}
          y={r * DOT_SPACING + (DOT_SPACING - DOT_SIZE) / 2}
          width={DOT_SIZE}
          height={DOT_SIZE}
          color={DOT_FG_COLOR}
        />,
      );
    }
  }
  return <Group>{rects}</Group>;
}

/**
 * Build a single DOT_SPACING × DOT_SPACING tile and return it as an
 * SkImage suitable for ImageShader repeat-tiling.
 */
function buildDotTile(): SkImage | null {
  const surface = Skia.Surface.MakeOffscreen(DOT_SPACING, DOT_SPACING);
  if (!surface) return null;
  const canvas = surface.getCanvas();
  canvas.clear(Skia.Color(DOT_BG_COLOR));
  const paint = Skia.Paint();
  paint.setColor(Skia.Color(DOT_FG_COLOR));
  paint.setAntiAlias(true);
  const cx = (DOT_SPACING - DOT_SIZE) / 2 + DOT_SIZE / 2;
  canvas.drawCircle(cx, cx, DOT_SIZE / 2, paint);
  surface.flush();
  const img = surface.makeImageSnapshot();
  if (typeof surface.dispose === 'function') surface.dispose();
  return img;
}

/**
 * Paint the dot background into the off-screen surface (called once at
 * surface init). Mirrors the live <DotPattern> so empty regions show
 * the same fill whether they're in the surface or behind it.
 */
function paintDotBackground(
  canvas: SkCanvas,
  width: number,
  height: number,
): void {
  const bg = Skia.Paint();
  bg.setColor(Skia.Color(DOT_BG_COLOR));
  canvas.drawRect(Skia.XYWHRect(0, 0, width, height), bg);
  const fg = Skia.Paint();
  fg.setColor(Skia.Color(DOT_FG_COLOR));
  fg.setAntiAlias(true);
  for (let y = 0; y < height; y += DOT_SPACING) {
    for (let x = 0; x < width; x += DOT_SPACING) {
      canvas.drawCircle(
        x + DOT_SPACING / 2,
        y + DOT_SPACING / 2,
        DOT_SIZE / 2,
        fg,
      );
    }
  }
}

/**
 * Decode a StripCheckpoint into an SkImage and stamp it into the
 * surface canvas at the angular x-offset implied by `strip.coverage`.
 * The source bytes (`strip.imageData`) are referenced only inside this
 * function — the caller may discard them as soon as we return.
 *
 * `rgbaBufferRef` is an optional caller-owned reusable RGBA scratch
 * buffer; passed through to `makeImageFromRgb` so the widening pass
 * doesn't allocate per-strip.
 */
function paintStripIntoSurface(
  canvas: SkCanvas,
  strip: StripCheckpoint,
  panoramaWidth: number,
  rgbaBufferRef?: React.MutableRefObject<Uint8Array | null>,
): void {
  const image = makeImageFromRgb(
    strip.imageData,
    strip.width,
    strip.height,
    rgbaBufferRef,
  );
  if (!image) return;
  try {
    const x = coverageToStripX(strip.coverage, panoramaWidth, strip.width);
    const dst = Skia.XYWHRect(x, 0, strip.width, strip.height);
    const src = Skia.XYWHRect(0, 0, strip.width, strip.height);
    const paint = Skia.Paint();
    paint.setAntiAlias(true);
    canvas.drawImageRect(image, src, dst, paint);
  } finally {
    if (typeof image.dispose === 'function') image.dispose();
  }
}

/**
 * Build an SkImage from tightly-packed RGB-uint8 bytes. Skia wants
 * RGBA, so we widen with an opaque alpha channel before handing the
 * bytes to MakeImage.
 *
 * If `rgbaBufferRef` is supplied, the widened bytes are written into
 * the caller's reusable buffer instead of a freshly allocated one
 * (eliminates ~11.8MB of GC churn over a 36-strip scan; see comment
 * by `rgbaBufferRef` in PanoramaCanvas). The buffer is grown lazily
 * if a strip's pixel count exceeds its current size — rare in practice
 * since STRIP_WIDTH × height is fixed at 80×1024.
 */
function makeImageFromRgb(
  rgb: Uint8Array,
  width: number,
  height: number,
  rgbaBufferRef?: React.MutableRefObject<Uint8Array | null>,
): SkImage | null {
  const needed = width * height * 4;
  let rgba: Uint8Array;
  if (rgbaBufferRef) {
    if (
      rgbaBufferRef.current === null ||
      rgbaBufferRef.current.length < needed
    ) {
      rgbaBufferRef.current = new Uint8Array(needed);
    }
    rgba = rgbaBufferRef.current;
  } else {
    rgba = new Uint8Array(needed);
  }
  for (let i = 0, j = 0; i < rgb.length; i += 3, j += 4) {
    rgba[j] = rgb[i];
    rgba[j + 1] = rgb[i + 1];
    rgba[j + 2] = rgb[i + 2];
    rgba[j + 3] = 255;
  }
  // Skia.Data.fromBytes copies the bytes immediately, so reusing the
  // source buffer for the next strip is safe. If we ever need a
  // sub-range of the buffer (when the cached array is larger than
  // `needed`), pass a typed-array view rather than the full buffer.
  const view =
    rgba.length === needed ? rgba : rgba.subarray(0, needed);
  const data = Skia.Data.fromBytes(view);
  try {
    // 1.x: Skia.Image.MakeImage(info, data, rowBytes). The info object
    // describes width/height/colorType/alphaType.
    return Skia.Image.MakeImage(
      {
        width,
        height,
        colorType: ColorType.RGBA_8888,
        alphaType: AlphaType.Opaque,
      },
      data,
      width * 4,
    );
  } finally {
    if (typeof data.dispose === 'function') data.dispose();
  }
}

const styles = StyleSheet.create({
  container: {
    overflow: 'hidden',
    backgroundColor: DOT_BG_COLOR,
  },
  legendWrap: {
    ...StyleSheet.absoluteFillObject,
    alignItems: 'center',
    justifyContent: 'center',
    paddingHorizontal: 16,
  },
  legendText: {
    fontSize: 12,
    fontWeight: '500',
    letterSpacing: 0.2,
    color: colors.scanInkFaint,
    textAlign: 'center',
  },
});
