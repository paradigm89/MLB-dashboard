/**
 * RunDistributionChart — shows the P10/median/P90 run range for each team
 * as a simple range visualization.
 */

interface RunDist {
  median: number;
  p10: number;
  p90: number;
}

interface Props {
  homeAbbr: string;
  awayAbbr: string;
  homeRuns?: RunDist;
  awayRuns?: RunDist;
}

export default function RunDistributionChart({ homeAbbr, awayAbbr, homeRuns, awayRuns }: Props) {
  if (!homeRuns || !awayRuns) return null;

  const data = [
    { team: awayAbbr, p10: awayRuns.p10, median: awayRuns.median, p90: awayRuns.p90 },
    { team: homeAbbr, p10: homeRuns.p10, median: homeRuns.median, p90: homeRuns.p90 },
  ];

  return (
    <div>
      <h4 className="title">Expected Run Totals (P10 – Median – P90)</h4>
      <div className="chart-container">
        {data.map((d) => (
          <div key={d.team} className="team-row">
            <span className="team-label">{d.team}</span>
            <div className="range-bar">
              <span className="range-label lo">{d.p10}</span>
              <div className="bar" style={{ width: `${d.median * 8}%` }}>
                <span className="median-val">{d.median}</span>
              </div>
              <span className="range-label hi">{d.p90}</span>
            </div>
          </div>
        ))}
      </div>

      <style jsx>{`
        .title { font-size: 14px; color: #ccc; margin: 0 0 12px; }
        .chart-container { display: flex; flex-direction: column; gap: 10px; }
        .team-row { display: flex; align-items: center; gap: 10px; }
        .team-label { width: 40px; font-weight: 700; color: #fff; font-size: 13px; }
        .range-bar { display: flex; align-items: center; gap: 6px; flex: 1; }
        .bar {
          background: #1565c0;
          height: 22px;
          border-radius: 4px;
          min-width: 30px;
          display: flex;
          align-items: center;
          justify-content: center;
          transition: width 0.3s ease;
        }
        .median-val { font-size: 12px; font-weight: 600; color: #fff; }
        .range-label { font-size: 11px; color: #888; width: 18px; }
        .lo { text-align: right; }
        .hi { text-align: left; }
      `}</style>
    </div>
  );
}
