/**
 * BatterMatchupTable — lineup batter matchup xwOBA scores vs. opposing SP.
 * Green = strong vs. opponent pitcher; red = weak.
 */
import { BatterMatchup } from "@/lib/api";

interface Props {
  batters?: BatterMatchup[];
  teamAbbr: string;
  vsSpName?: string;
}

function xwobaColor(xwoba?: number): string {
  if (xwoba == null) return "#666";
  if (xwoba >= 0.380) return "#4caf50";
  if (xwoba >= 0.330) return "#8bc34a";
  if (xwoba >= 0.300) return "#fff";
  if (xwoba >= 0.270) return "#ff9800";
  return "#f44336";
}

function fmt(v?: number) {
  return v != null ? v.toFixed(3) : "—";
}

export default function BatterMatchupTable({ batters, teamAbbr, vsSpName }: Props) {
  if (!batters || batters.length === 0) {
    return <div className="empty">No batter matchup data</div>;
  }

  const sorted = [...batters].sort((a, b) => a.batting_order - b.batting_order);

  return (
    <div>
      <h4 className="title">
        {teamAbbr} Lineup{vsSpName ? ` vs. ${vsSpName}` : ""}
      </h4>
      <table className="table">
        <thead>
          <tr>
            <th>#</th>
            <th>Player ID</th>
            <th title="Projected xwOBA vs. this pitcher (season avg blended with matchup history)">xwOBA</th>
            <th title="Sample size: career plate appearances vs. this pitcher">PA vs. SP</th>
          </tr>
        </thead>
        <tbody>
          {sorted.map((b) => (
            <tr key={b.batting_order}>
              <td className="pos">{b.batting_order}</td>
              <td>{b.batter_id}</td>
              <td style={{ color: xwobaColor(b.projected_xwoba), fontWeight: 600 }}>
                {fmt(b.projected_xwoba)}
              </td>
              <td className="pa">{b.sample_pa > 0 ? b.sample_pa : "—"}</td>
            </tr>
          ))}
        </tbody>
      </table>

      <style jsx>{`
        .title { font-size: 14px; color: #ccc; margin: 0 0 8px; }
        .table { width: 100%; border-collapse: collapse; font-size: 13px; }
        th { text-align: left; color: #888; padding: 4px 8px; border-bottom: 1px solid #333; font-weight: 400; }
        td { padding: 4px 8px; color: #ddd; }
        tr:hover td { background: #1a2a4a; }
        .pos { color: #888; width: 24px; }
        .pa { color: #666; }
        .empty { color: #555; font-size: 13px; }
      `}</style>
    </div>
  );
}
