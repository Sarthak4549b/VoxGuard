import Database from 'better-sqlite3';
import { logger } from '../utils/logger';

let db: Database.Database | null = null;

export function initDB(): Database.Database {
  const dbPath = process.env.DB_PATH || './voxguard.db';

  const instance = new Database(dbPath);
  instance.pragma('journal_mode = WAL');
  instance.pragma('foreign_keys = ON');

  instance.exec(`
    CREATE TABLE IF NOT EXISTS sessions (
      id TEXT PRIMARY KEY,
      created_at TEXT NOT NULL,
      ended_at TEXT,
      status TEXT NOT NULL DEFAULT 'active'
    );

    CREATE TABLE IF NOT EXISTS chunk_analyses (
      id INTEGER PRIMARY KEY AUTOINCREMENT,
      session_id TEXT NOT NULL,
      chunk_id INTEGER NOT NULL,
      spoof_score REAL,
      speaker_similarity REAL,
      speaker_status TEXT,
      spoof_label TEXT,
      transcript TEXT,
      llm_risk_score REAL,
      detected_signals TEXT,
      final_risk_score REAL,
      risk_level TEXT,
      reason TEXT,
      latency_ms INTEGER,
      timestamp TEXT,
      FOREIGN KEY (session_id) REFERENCES sessions(id)
    );
  `);

  db = instance;
  logger.info('Database initialized');
  return instance;
}

export function getDB(): Database.Database {
  if (!db) {
    throw new Error('Database not initialized. Call initDB() first.');
  }
  return db;
}
