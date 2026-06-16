import type { UmapCoord } from "./types";

export default function PossibilityMap({ coords }: { coords: UmapCoord[] }) {
  if (coords.length === 0) {
    return (
      <div className="flex items-center justify-center h-40 text-gray-600 text-sm">
        No data yet
      </div>
    );
  }

  const xs = coords.map((c) => c.x);
  const ys = coords.map((c) => c.y);
  const xMin = Math.min(...xs);
  const xMax = Math.max(...xs);
  const yMin = Math.min(...ys);
  const yMax = Math.max(...ys);
  const pad = 0.5;

  const toSvg = (x: number, y: number) => {
    const sx = ((x - xMin + pad) / (xMax - xMin + 2 * pad)) * 100;
    const sy = ((y - yMin + pad) / (yMax - yMin + 2 * pad)) * 100;
    return { sx, sy };
  };

  return (
    <div className="bg-surface border border-border rounded p-4">
      <svg viewBox="0 0 100 100" className="w-full aspect-square">
        {coords.map((c, i) => {
          const { sx, sy } = toSvg(c.x, c.y);
          return (
            <g key={i}>
              <circle cx={sx} cy={sy} r={2.5} fill="#00ff88" opacity={0.8} />
              <text
                x={sx + 3}
                y={sy + 1}
                fontSize={3}
                fill="#999"
                fontFamily="JetBrains Mono"
              >
                {c.config_label}
              </text>
            </g>
          );
        })}
      </svg>
    </div>
  );
}
