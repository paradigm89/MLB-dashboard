/**
 * Typed API client for the MLB prediction backend.
 * All fetch calls go through this module so the base URL is configured once.
 */

const API_BASE = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

export interface TeamInfo { team_id: number; team_abbr: string }
export interface PitcherInfo { pitcher_id: number; pitcher_name?: string; throws?: string }
export interface RunDistribution { median: number; p10: number; p90: number }

export interface GamePrediction {
  game_pk: number;
  game_date: string;
  home_team: TeamInfo;
  away_team: TeamInfo;
  home_sp?: PitcherInfo;
  away_sp?: PitcherInfo;
  win_prob_home?: number;
  win_prob_away?: number;
  runs_home?: RunDistribution;
  runs_away?: RunDistribution;
  home_lineup_xwoba?: number;
  away_lineup_xwoba?: number;
  lineup_edge?: number;
  home_lineup_confirmed: boolean;
  away_lineup_confirmed: boolean;
  simulation_method: string;
  prediction_generated_at?: string;
}

export interface GameScheduleItem {
  game_pk: number;
  game_date: string;
  time_et?: string;
  venue_name?: string;
  home_team: TeamInfo;
  away_team: TeamInfo;
  home_sp?: PitcherInfo;
  away_sp?: PitcherInfo;
  home_lineup_confirmed: boolean;
  away_lineup_confirmed: boolean;
  predictions?: GamePrediction;
  data_as_of?: string;
}

export interface TodayScheduleResponse {
  date: string;
  games: GameScheduleItem[];
  total_games: number;
  lineups_confirmed: number;
  lineups_projected: number;
  last_refreshed?: string;
}

export interface SystemStatus {
  last_morning_refresh: { ran_at?: string; status?: string; games_updated?: number; error?: string };
  last_afternoon_refresh: { ran_at?: string; status?: string; lineups_confirmed?: number; error?: string };
  last_retrain: { ran_at?: string; status?: string; error?: string };
  db_last_updated?: string;
  active_model_versions: Array<{
    model_type: string;
    trained_at?: string;
    brier_score?: number;
    mae_runs?: number;
  }>;
  active_errors: Array<{ type: string; message: string; occurred_at?: string }>;
}

async function get<T>(path: string): Promise<T> {
  const res = await fetch(`${API_BASE}${path}`, { cache: "no-store" });
  if (!res.ok) throw new Error(`API ${path} returned ${res.status}`);
  return res.json() as Promise<T>;
}

export const api = {
  getTodaySchedule: () => get<TodayScheduleResponse>("/api/schedule/today"),
  getGamePrediction: (gamePk: number) =>
    get<GamePrediction>(`/api/predict/game?game_pk=${gamePk}`),
  getSystemStatus: () => get<SystemStatus>("/api/system/status"),
  getTeamStats: (teamId: number, season?: number) =>
    get(`/api/stats/team/${teamId}${season ? `?season=${season}` : ""}`),
  getPlayerStats: (playerId: number, statType: "batting" | "pitching" = "batting") =>
    get(`/api/stats/player/${playerId}?stat_type=${statType}`),
};
