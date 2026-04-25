export interface SessionSummary {
  session_id: number;
  session_type: "diagnostic" | "reading";
  status: string;
  started_at: string;
  ended_at: string | null;
  accuracy_pct: number | null;
  speed_wpm: number | null;
}

export interface StudentProfile {
  student_id: number;
  email: string;
  reading_level: string | null;
  avg_speed_wpm: number;
  avg_accuracy_pct: number;
  attention_score: number;
  difficult_words: string[];
  model_profile_scores: Record<string, number>;
  recent_sessions: SessionSummary[];
}

export interface ProgressEntry {
  id: number;
  session_id: number;
  accuracy_trend: number;
  words_practiced: string[];
  timestamp: string;
}

export interface StudentProgress {
  student_id: number;
  entries: ProgressEntry[];
  difficult_words: string[];
}

export interface TeacherStudentSummary {
  student_id: number;
  name: string;
  email: string;
  reading_level: string | null;
  avg_accuracy_pct: number;
  avg_speed_wpm: number;
  attention_score: number;
  difficult_words: string[];
  last_session_date: string | null;
}
