import createQrCode from 'qrcode-generator';
import { useMemo } from 'react';
import Svg, { Path, Rect } from 'react-native-svg';

const QUIET_ZONE_MODULES = 4;
const LIGHT = '#FFFFFF';
const DARK = '#000000';

type Props = {
  value: string;
  size: number;
  label: string;
};

/** Stays black on white in the dark theme too, because many scanners can't read it inverted. */
export const QrCode = ({ value, size, label }: Props) => {
  const { modules, path } = useMemo(() => {
    const code = createQrCode(0, 'M');
    code.addData(value);
    code.make();
    const count = code.getModuleCount();
    let cells = '';
    for (let row = 0; row < count; row += 1) {
      for (let column = 0; column < count; column += 1) {
        if (code.isDark(row, column)) cells += `M${column} ${row}h1v1h-1z`;
      }
    }
    return { modules: count, path: cells };
  }, [value]);
  const extent = modules + QUIET_ZONE_MODULES * 2;

  return (
    <Svg
      width={size}
      height={size}
      viewBox={`${-QUIET_ZONE_MODULES} ${-QUIET_ZONE_MODULES} ${extent} ${extent}`}
      accessibilityRole="image"
      accessibilityLabel={label}
    >
      <Rect
        x={-QUIET_ZONE_MODULES}
        y={-QUIET_ZONE_MODULES}
        width={extent}
        height={extent}
        fill={LIGHT}
      />
      <Path d={path} fill={DARK} />
    </Svg>
  );
};
