/**
 * Renders a captured image with one or more colored bounding-box
 * rectangles drawn on top.
 *
 * The captured image's intrinsic dimensions live in scanStore; the
 * `bbox` values are in that pixel space. Because we render with
 * `resizeMode="contain"`, the displayed image is letterboxed inside
 * the container — so the overlay needs to translate image-pixel
 * coordinates into container-relative coordinates using the same
 * letterbox math.
 */

import React, { useMemo, useState } from 'react';
import {
  Image,
  LayoutChangeEvent,
  StyleSheet,
  View,
  type StyleProp,
  type ViewStyle,
} from 'react-native';

import type { BBox, RuleStatus } from '@src/api/types';
import { colors, radius } from '@src/theme';

export interface BBoxOverlayItem {
  bbox: BBox;
  status: RuleStatus;
  // Optional id for React keys when multiple boxes are drawn.
  id?: string;
}

export interface BBoxOverlayProps {
  // Image source dimensions and uri. Width / height are the captured
  // image's pixel dimensions (used to compute the contain letterbox).
  image: { uri: string; width: number; height: number };
  items: ReadonlyArray<BBoxOverlayItem>;
  // Override the default aspectRatio (defaults to 0.65 — the same
  // portrait-ish ratio used for the report thumbnail strip).
  style?: StyleProp<ViewStyle>;
}

export function BBoxOverlay({
  image,
  items,
  style,
}: BBoxOverlayProps): React.ReactElement {
  const [container, setContainer] = useState<{ width: number; height: number } | null>(
    null
  );

  const onLayout = (e: LayoutChangeEvent) => {
    const { width, height } = e.nativeEvent.layout;
    setContainer({ width, height });
  };

  // Compute object-fit: contain letterbox geometry. The rendered image
  // preserves aspect ratio, so it's scaled by the smaller of the two
  // ratios and centered along the other axis.
  const letterbox = useMemo(() => {
    if (!container) return null;
    const imgAspect = image.width / image.height;
    const boxAspect = container.width / container.height;
    let renderedW: number;
    let renderedH: number;
    if (imgAspect > boxAspect) {
      // Image is wider than container — full width, letterboxed top/bottom.
      renderedW = container.width;
      renderedH = container.width / imgAspect;
    } else {
      // Image is taller than container — full height, letterboxed sides.
      renderedH = container.height;
      renderedW = container.height * imgAspect;
    }
    return {
      offsetX: (container.width - renderedW) / 2,
      offsetY: (container.height - renderedH) / 2,
      scale: renderedW / image.width,
    };
  }, [container, image.width, image.height]);

  return (
    <View style={[styles.box, style]} onLayout={onLayout}>
      <Image
        source={{ uri: image.uri }}
        style={styles.image}
        resizeMode="contain"
      />
      {letterbox
        ? items.map((item, idx) => {
            const [bx, by, bw, bh] = item.bbox;
            return (
              <View
                key={item.id ?? idx}
                pointerEvents="none"
                style={[
                  styles.frame,
                  {
                    left: letterbox.offsetX + bx * letterbox.scale,
                    top: letterbox.offsetY + by * letterbox.scale,
                    width: bw * letterbox.scale,
                    height: bh * letterbox.scale,
                    borderColor: bboxColorFor(item.status),
                  },
                ]}
              />
            );
          })
        : null}
    </View>
  );
}

function bboxColorFor(status: RuleStatus): string {
  switch (status) {
    case 'pass':
      return colors.pass;
    case 'fail':
      return colors.fail;
    case 'advisory':
      return colors.advisory;
  }
}

const styles = StyleSheet.create({
  box: {
    aspectRatio: 0.65,
    backgroundColor: colors.surfaceAlt,
    borderColor: colors.border,
    borderWidth: 1,
    borderRadius: radius.sm,
    overflow: 'hidden',
  },
  image: {
    width: '100%',
    height: '100%',
  },
  frame: {
    position: 'absolute',
    borderWidth: 2,
    borderRadius: 2,
  },
});
