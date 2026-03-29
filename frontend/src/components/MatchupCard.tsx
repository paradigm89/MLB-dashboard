/**
 * MatchupCard — game card shown on the today's games dashboard.
 * Links to the /game/[id] detail page.
 */
import Link from "next/link";
import WinProbBar from "./WinProbBar";
import { GameScheduleItem } from "@/lib/api";

interface Props {
  game: GameScheduleItem;
}

export default function MatchupCard({ game }: Props) {
  const pred = game.predictions;
  const hasPrediction = pred != null;

  const lineupEdge = pred?.lineup_edge;
  const edgeTeam = lineupEdge != null
    ? (lineupEdge > 0 ? game.home_team.team_abbr : game.away_team.team_abbr)
    : null;
  const edgeMag = lineupEdge != null ? Math.abs(lineupEdge).toFixed(3) : null;

  return (
    <Link href={`/game/${game.game_pk}`} className="card-link">
      <div className="card">
        {/* Header */}
        <div className="header">
          <div className="teams">
            <span className="team away">{game.away_team.team_abbr}</span>
            <span className="at">@</span>
            <span className="team home">{game.home_team.team_abbr}</span>
          </div>
          {game.venue_name && <span className="venue">{game.venue_name}</span>}
        </div>

        {/* Pitching matchup */}
        <div className="pitchers">
          <span>{game.away_sp ? (game.away_sp.pitcher_name ?? `#${game.away_sp.pitcher_id}`) : "SP TBD"}</span>
          <span className="sep">vs</span>
          <span>{game.home_sp ? (game.home_sp.pitcher_name ?? `#${game.home_sp.pitcher_id}`) : "SP TBD"}</span>
        </div>

        {/* Win probability bar */}
        {hasPrediction ? (
          <WinProbBar
            awayAbbr={game.away_team.team_abbr}
            homeAbbr={game.home_team.team_abbr}
            winProbHome={pred!.win_prob_home}
            winProbAway={pred!.win_prob_away}
            homeConfirmed={game.home_lineup_confirmed}
            awayConfirmed={game.away_lineup_confirmed}
            generatedAt={game.data_as_of}
          />
        ) : (
          <div className="no-pred">Prediction unavailable</div>
        )}

        {/* Lineup edge */}
        {edgeTeam && (
          <div className="edge">
            Lineup edge: <strong>{edgeTeam}</strong> (+{edgeMag} xwOBA)
          </div>
        )}

        {/* Run totals */}
        {pred?.runs_home && pred?.runs_away && (
          <div className="runs">
            <span>{game.away_team.team_abbr}: {pred.runs_away.median.toFixed(1)} R</span>
            <span className="sep">|</span>
            <span>{game.home_team.team_abbr}: {pred.runs_home.median.toFixed(1)} R</span>
          </div>
        )}
      </div>

      <style jsx>{`
        .card-link { text-decoration: none; color: inherit; }
        .card {
          background: #16213e;
          border: 1px solid #333;
          border-radius: 10px;
          padding: 16px;
          cursor: pointer;
          transition: border-color 0.2s, transform 0.1s;
        }
        .card:hover { border-color: #1565c0; transform: translateY(-2px); }
        .header { display: flex; justify-content: space-between; align-items: baseline; margin-bottom: 8px; }
        .teams { font-size: 18px; font-weight: 700; }
        .team { color: #fff; }
        .at { color: #666; margin: 0 6px; }
        .venue { font-size: 11px; color: #666; }
        .pitchers { font-size: 12px; color: #888; margin-bottom: 8px; display: flex; gap: 6px; }
        .sep { color: #555; }
        .no-pred { font-size: 13px; color: #555; padding: 8px 0; }
        .edge { font-size: 12px; color: #90caf9; margin-top: 6px; }
        .runs { font-size: 12px; color: #aaa; display: flex; gap: 8px; margin-top: 4px; }
      `}</style>
    </Link>
  );
}
