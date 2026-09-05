import { randomUUID } from 'crypto';
import { Router } from 'express';
import { createSession, endSession, getSessionHistory } from '../db/queries';

const router = Router();

// POST /sessions — create a new session
router.post('/', (_req, res) => {
  const sessionId = randomUUID();
  const createdAt = new Date().toISOString();
  createSession(sessionId);
  res.json({ session_id: sessionId, created_at: createdAt, status: 'active' });
});

// GET /sessions/history — recent sessions with their chunk analyses
router.get('/history', (req, res) => {
  const raw = Number(req.query.limit);
  let limit = Number.isFinite(raw) ? Math.floor(raw) : 20;
  if (limit < 1) limit = 20;
  if (limit > 100) limit = 100;

  const sessions = getSessionHistory(limit);
  res.json({ sessions });
});

// DELETE /sessions/:sessionId — end a session
router.delete('/:sessionId', (req, res) => {
  const { sessionId } = req.params;
  endSession(sessionId);
  res.json({ session_id: sessionId, status: 'ended' });
});

export default router;
