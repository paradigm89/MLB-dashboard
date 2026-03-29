/**
 * WinProbBar — horizontal probability bar for a two-team matchup.
 * Shows home team on left, away on right, with width proportional to win prob.
 * Greys out and shows "Lineup TBD" badge when lineup is not yet confirmed.
 */
interface Props {
  homeAbbr: string;
  awayAbbr: string;
  winProbHome?: number;
  winProbAway?: number;
  homeConfirmed: boolean;
  awayConfirmed: boolean;
  generatedAt?: string;
}

export default function WinProbBar({
  homeAbbr,
  awayAbbr,
  winProbHome,
  winProbAway,
  homeConfirmed,
  awayConfirmed,
  generatedAt,
}: Props) {
  const hasProb = winProbHome != null && winProbAway != null;
  const homePct = hasProb ? Math.round(winProbHome! * 100) : 50;
  const awayPct = hasProb ? Math.round(winProbAway! * 100) : 50;
  const bothConfirmed = homeConfirmed && awayConfirmed;

  return (
    <div className="container">
      <div className="labels">
        <span className="team home-label">{awayAbbr}</span>
        <span className="pct-label">{awayPct}%</span>
        <span className="vs">vs</span>
        <span className="pct-label">{homePct}%</span>
        <span className="team home-label">{homeAbbr}</span>
      </div>

      <div className="bar-track">
        <div className="bar-away" style={{ width: `${awayPct}%` }} />
        <div className="bar-home" style={{ width: `${homePct}%` }} />
      </div>

      <div className="footer">
        {!awayConfirmed && <span className="badge projected">{awayAbbr} lineup projected</span>}
        {!homeConfirmed && <span className="badge projected">{homeAbbr} lineup projected</span>}
        {generatedAt && (
          <span className="ts">
            Updated {new Date(generatedAt.endsWith("Z") ? generatedAt : generatedAt + "Z").toLocaleTimeString("en-US", {
              hour: "numeric",
              minute: "2-digit",
            })}
          </span>
        )}
      </div>

      <style jsx>{`
        .container { padding: 8px 0; }
        .labels {
          display: flex;
          justify-content: space-between;
          align-items: center;
          font-size: 13px;
          color: #ccc;
          margin-bottom: 4px;
        }
        .team { font-weight: 700; font-size: 14px; color: #fff; }
        .pct-label { font-size: 15px; font-weight: 600; color: #fff; }
        .vs { color: #666; font-size: 11px; }
        .bar-track {
          display: flex;
          height: 10px;
          border-radius: 5px;
          overflow: hidden;
          background: #222;
        }
        .bar-away { background: #e53935; transition: width 0.4s ease; }
        .bar-home { background: #1565c0; transition: width 0.4s ease; }
        .footer {
          display: flex;
          gap: 8px;
          align-items: center;
          margin-top: 4px;
          flex-wrap: wrap;
        }
        .badge {
          font-size: 10px;
          padding: 2px 6px;
          border-radius: 4px;
          font-weight: 600;
        }
        .projected { background: #e65100; color: #fff; }
        .ts { font-size: 11px; color: #666; margin-left: auto; }
      `}</style>
    </div>
  );
}
