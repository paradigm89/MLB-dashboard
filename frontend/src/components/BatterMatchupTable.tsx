/**
 * BatterMatchupTable — sortable table of lineup batter matchup xwOBA scores.
 * Green = strong vs. opponent pitcher; red = weak.
 */
interface BatterScore {
  pos: number;
  batter_id: number;
  xwoba: number;
}

interface Props {
  batters: BatterScore[];
  teamAbbr: string;
  vsSpName?: string;
}

function xwobaColor(xwoba: number): string {
  if (xwoba >= 0.380) return "#4caf50";
  if (xwoba >= 0.330) return "#8bc34a";
  if (xwoba >= 0.300) return "#fff";
  if (xwoba >= 0.270) return "#ff9800";
  return "#f44336";
}

export default function BatterMatchupTable({ batters, teamAbbr, vsSpName }: Props) {
  if (!batters || batters.length === 0) {
    return <div className="empty">No batter matchup data</div>;
  }

  const sorted = [...batters].sort((a, b) => a.pos - b.pos);

  return (
    <div>
      <h4 className="title">
        {teamAbbr} Lineup{vsSpName ? ` vs. ${vsSpName}` : ""}
      </h4>
      <table className="table">
        <thead>
          <tr>
            <th>#</th>
            <th>Batter ID</th>
            <th>xwOBA</th>
          </tr>
        </thead>
        <tbody>
          {sorted.map((b) => (
            <tr key={b.pos}>
              <td className="pos">{b.pos}</td>
              <td>{b.batter_id}</td>
              <td style={{ color: xwobaColor(b.xwoba), fontWeight: 600 }}>
                {b.xwoba.toFixed(3)}
              </td>
            </tr>
          ))}
        </tbody>
      </table>

      <style jsx>{`
        .title { font-size: 14px; color: #ccc; margin: 0 0 8px; }
        .table { width: 100%; border-collapse: collapse; font-size: 13px; }
        th { text-align: left; color: #888; padding: 4px 8px; border-bottom: 1px solid #333; }
        td { padding: 4px 8px; color: #ddd; }
        tr:hover td { background: #1a2a4a; }
        .pos { color: #888; }
        .empty { color: #555; font-size: 13px; }
      `}</style>
    </div>
  );
}
